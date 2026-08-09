# Research dossier — Aletheia-NVFP4

State of the art as of **August 2026**, and how each finding maps to a concrete decision in
[`Aletheia_NVFP4_Pretrain.ipynb`](Aletheia_NVFP4_Pretrain.ipynb). Everything below was read during
this design pass; nothing is recalled from model memory without a source. Where sources disagree,
both are recorded and the notebook exposes a flag rather than silently picking a winner.

Contents:

1. [Starting point: what already exists in the Aletheia repos](#1-starting-point)
2. [4-bit pretraining](#2-4-bit-pretraining)
3. [Hardware and software reality check](#3-hardware-and-software-reality-check)
4. [Tokenizer](#4-tokenizer)
5. [Architecture](#5-architecture)
6. [Optimizer and schedule](#6-optimizer-and-schedule)
7. [Token budget: why Chinchilla is ignored](#7-token-budget)
8. [Data](#8-data)
9. [Evaluation](#9-evaluation)
10. [Open risks](#10-open-risks)
11. [Source index](#11-source-index)

---

## 1. Starting point

`Aletheia-NVFP4` is a **new model**: new tokenizer, new architecture, random init, no distillation
and no weights from anywhere. This section is therefore not a baseline to extend — it is the local
*evidence base*, four repositories' worth of measurements that constrain the design choices in §4–§8.

The distinction that matters most: **`aletheia-lm` is a code model, this is a chat model.** They
share a name and a domain, nothing else. `aletheia-lm` is also **private** — an unauthenticated
fetch 404s, which is why an earlier pass of this dossier wrongly recorded it as nonexistent.

### `hotocoo/aletheia` — the knowledge domain

A from-scratch, AI-native operating system in Rust (plus C/C++/assembly), targeting AMD64 and RISC-V
first-class with ARM64 for bootstrap. It treats AI as *a native but untrusted collaborator*: the model
proposes plans, and the OS validates, authorizes and executes them deterministically. The pipeline is
`Intent → context building → model proposal → validation → capability evaluation → policy approval →
execution → verification → provenance recording`.

Subsystems that the assistant model must be able to talk about:

- domain primitives: Entity, Capability, Context, Intent, Action, Memory, Relationship
- capabilities engine — unforgeable, attenuable, revocable authority
- policy engine — governance *separate from* authority, requiring human approval
- Context Fabric — capability-aware structured retrieval, explicitly **not** RAG
- WASM component runtime with no ambient authority
- `no_std` microkernel across three CPU architectures
- docs tree: PRD, SAD, ADRs, STATUS.md

**Consequences for the notebook.** These identifiers become tokenizer special tokens
(`<|intent|>`, `<|capability|>`, `<|policy|>`, `<|provenance|>`, …), the repo is upsampled ~24× in the
data mix, the synthetic anneal set is generated from its ADRs/docs/Rust items, and the closed-book
knowledge probe in §11a asserts on exactly these facts.

### `hotocoo/aletheia1Bmx` — the predecessor training notebook

`aletheiaMLX.ipynb`, MLX on Apple Silicon: 1B-class Llama-style decoder — `d_model=2048`, 24 layers,
16 heads / 4 KV heads (GQA), `intermediate_size=5632`, SwiGLU, RMSNorm(1e-5), non-traditional RoPE,
`seq_len=2048`, micro-batch 1 × grad-accum 32, AdamW(0.9, 0.95), wd 0.1, LR 1e-4 with 500-step warmup
+ cosine, clip 0.5, 5000 steps, `mx.bfloat16`. Data: local clones of OS/systems repos tokenized into
`uint32` memmap shards of 500M tokens; chunked cross-entropy (512 rows at a time) to survive the 64k
vocab logit tensor; checkpoint manifest with `keep=2`.

**Reused — engineering patterns only, never weights:** the memmap shard layout (now `uint16` plus a
document index), the checkpoint manifest, and the telemetry/plot structure. Its chunked
cross-entropy is *not* reused; FP32 logits with a z-loss make it unnecessary.
**Everything else differs:** MLX→CUDA/TE, bfloat16→NVFP4, cosine→warmup-hold-decay, AdamW→Muon+AdamW,
code-only data→chat-first mixture, and a `LlamaTokenizer(vocab_file=...)` shim→a tokenizer this
notebook trains itself.

### `hotocoo/aletheia-lm` (private) — the code model, and the measurements worth borrowing

A production MLX stack: `src/aletheia_lm/{config,data,hf_ingest,model,tokenizer_train,train}.py`,
plus `docs/PRETRAIN-SOTA-2026.md` and `docs/TOKENIZER-V3.md`. Its model is 280.8M params
(248.0M non-embedding), 1024×22, GQA 16/4, SwiGLU 2816, QK-norm, tied embeddings, z-loss 1e-4,
SWA 1024 at 3:1, RoPE θ=500k, Muon 2e-3 + AdamW 3e-4, warmup-hold-decay, and a live run at
500,000 steps / 8.192B tokens / 3,032 tok/s measured / 33 tokens per non-embedding parameter,
every token unique.

Four things from it are used here **as evidence, not as inheritance**:

1. **`TOKENIZER-V3.md`'s five-arm sweep** (2026-08-07, same class of corpus, byte-exact round-trip
   on every arm). Held-out bytes/token, higher is better:

   | arm | local | hub | comb@0.9 | tok/s | corpus B/s | params |
   |---|---:|---:|---:|---:|---:|---:|
   | `aletheia_tok_v2` 64k (shipped) | 3.7048 | 3.1370 | 3.1858 | — | — | 313.6M |
   | 64k v2 (mixture) — control | 3.7080 | 3.3179 | 3.3532 | 2193 | 7354 | 313.6M |
   | 32k v2 (mixture) | 3.5210 | 3.1653 | 3.1976 | 2408 | 7700 | 280.8M |
   | 64k `xw` (mixture) | 4.0100 | 3.6058 | 3.6425 | 2193 | 7988 | 313.6M |
   | **32k `xw` (mixture)** — adopted | 3.7612 | 3.3842 | 3.4185 | 2408 | **8232** | 280.8M |

   Three isolated effects, each measured: fitting the tokenizer to the actual *mixture* is free on
   the local half and +5.8% on the Hub half; **lifting the whitespace barrier is +8.6% at fixed
   vocabulary** — the largest single tokenizer effect, because systems code's most frequent bigrams
   (`\n    return`, `);\n`, `} else if (`) cross whitespace and a GPT-4-style regex forbids them;
   and halving 64k→32k costs 6.1% compression but buys 9.8% throughput, netting **+3.1% bytes of
   corpus per second**. Note `32k xw` beats `64k v2` outright. This is why §4 below trains a
   32k-class vocabulary with whitespace-crossing merges instead of the 65,536 an earlier draft used.
2. **The `xw` split regex itself**, reused verbatim as this notebook's stage-1 pretokenizer:
   `'(?i:[sdmt]|ll|ve|re)|(?i:0x…|0b…|0o…)|\s*[^\r\n\p{L}\p{N}]?\p{L}+|\s*\p{N}{1,3}|\s*[^\s\p{L}\p{N}]+[\r\n]*|\s+`
3. **`TOKENIZER-V3.md`'s own open item**: a *faithful* SuperBPE run is named as the largest untried
   tokenizer experiment there, blocked because "`tokenizers` 0.22 has no warm-start path for
   `BpeTrainer` … a faithful two-stage run needs a fork of the Rust trainer". §4 below unblocks it
   without a fork — stage 2 is learned over the stage-1 *id stream*, so the Rust trainer is never
   asked to warm-start.
4. **A measured defect worth not repeating**: `config.py` records that bf16 master weights (2⁻⁸
   relative resolution) forced `lr_min` up to 2e-4, so AdamW's decay phase spanned only 1.5× while
   Muon's spanned 10× — the embedding path effectively never got a decay. This notebook keeps FP32
   master weights with the cast inside the matmul, and `lr_min` returns to 3e-5.

### `hotocoo/aletheiatokenizer` — the first tokenizer

`aletheiacode64k`: SentencePiece **unigram**, `vocab_size=64000`, `byte_fallback=True`,
`character_coverage=1.0`, `normalization_rule_name="identity"`, `split_digits=False`,
`remove_extra_whitespaces=False`, `max_sentencepiece_length=64`, `input_sentence_size=25M`,
user-defined symbols `<|system|> <|user|> <|assistant|> <|tool|> <|memory|> <|code|>`, then converted
to `LlamaTokenizerFast`. Corpus: shallow clones of rust, rust-analyzer, tokio, servo, llvm-project,
gcc, linux, u-boot, xv6-riscv, serenity, qemu, edk2, ninja, CMake, make, ripgrep, curl.

**Two defects for a chat model:** (a) the corpus contains no natural-language prose, so conversational
text tokenizes badly; (b) unigram with whitespace pretokenization cannot produce whitespace-crossing
pieces. Both are fixed in §4.

---

## 2. 4-bit pretraining

### 2.1 The reference recipe — NVIDIA, arXiv:2509.25149

*Pretraining Large Language Models with NVFP4* (Sept 2025; the definitive validation run reported
May 2026). Validated on a **12B hybrid Mamba-Transformer at a 10T-token horizon** — the first
published multi-trillion-token 4-bit pretraining.

NVFP4 format: `x = x_e2m1 · s_block · s_global`, where `x_e2m1` is E2M1 (1 sign, 2 exponent, 1
mantissa; max magnitude ±6), `s_block` is an **FP8 E4M3** scale shared by **16 consecutive values**,
and `s_global` is a per-tensor **FP32** scale. Two-level block scaling is what separates NVFP4 from
MXFP4's single E8M0 per-32 scale.

The recipe, as four independent mechanisms:

| Mechanism | Exactly what | Where |
|---|---|---|
| Random Hadamard Transform | **16×16** matrix, one fixed random sign vector shared by all layers for the whole run, applied to **Wgrad inputs only** — not to fprop, not to dgrad | smooths outliers in the column-wise operand |
| Stochastic rounding | **gradients only** (both dgrad and wgrad); round-to-nearest-even for weights and activations | removes quantization bias in the expectation |
| 2D block scaling | weights get **16×16** 2D scaling so a tensor and its transpose are numerically identical; activations and gradients keep 1×16 | needed because linear-layer gradients use both orientations |
| Selective high precision | **16% of linear layers stay BF16**: the **first two blocks and the final eight**. Embeddings, output head, norms, non-linearities, attention and **optimizer states** are never FP4 | the sensitive extremes of the residual stream |

Results: validation loss tracks an FP8 baseline within **<1.5% relative** during stable training;
MMLU-Pro 5-shot **62.58%** vs **62.62%** FP8. Switching NVFP4→BF16 after **8.2T of 10T tokens (18% of
training)** narrows the relative loss gap from **1.5% → 0.5%**; a later switch still helps.

**Mapped to:** `NVFP4BlockScaling` defaults (RHT, stochastic rounding, 2D weights all on),
`cfg.bf16_first_blocks=2` / `cfg.bf16_last_blocks=4` (≈20% of a 30-layer model),
`cfg.bf16_switch_frac=0.82`, and the rule that embeddings/head/norms/attention use `nn.Linear` in
BF16 regardless of recipe.

### 2.2 The dissent — AMD + Penn State, arXiv:2605.09825

*Pretraining Large Language Models with MXFP4 on Native FP4 Hardware* (May 2026). Progressively
enables FP4 in Fprop → Dgrad → Wgrad on a full Llama-3.1-8B pretraining over C4, on native FP4
hardware rather than emulation.

- **Wgrad quantization is the dominant driver of convergence degradation.** FP4 in Fprop and Dgrad
  alone costs only a modest token overhead.
- **Stochastic rounding and *randomized* Hadamard rotations fail to stabilize training once Wgrad is
  quantized; *deterministic* Hadamard rotations consistently restore stable optimization.** This
  directly contradicts the "insufficient randomness" framing.
- End-to-end: **9–10% speedup over the FP8 baseline for 8–9% more tokens.**

This is the single most important caveat in the dossier: the headline FP4 speedup is ~10%, not 2–3×,
because non-GEMM work does not shrink. **Mapped to:** `cfg.deterministic_hadamard`, documented as the
first thing to try if loss spikes appear only after FP4 touches gradients, and honest speedup
expectations in the benchmark section rather than quoting arithmetic peak.

### 2.3 The frontier — arXiv:2607.04422

*Full-Stack FP4: Stable LLM Pretraining with Quantized Projections, Optimizers, and Attention*
(July 2026). Extends FP4 to optimizer moments and the attention path; embeddings, layer norms and
non-quantized activations stay BF16; per-token/per-channel dynamic scaling; stochastic rounding; BF16
warmup then a scheduled transition into full FP4. Reaches BF16-comparable perplexity.

**Mapped to:** `cfg.fp4_attention` flag, default **off**. Rationale recorded in the notebook: it
works, and it adds two more failure modes to a run that already carries one novel number format.

### 2.4 Transformer Engine — the implementation

Verified against TE source (`transformer_engine/common/recipe/__init__.py`, `pytorch/quantization.py`)
rather than from memory, because the recipe API has churned. As of TE 2.17–2.19:

```python
NVFP4BlockScaling(
    disable_rht: bool = env NVTE_NVFP4_DISABLE_RHT,
    disable_stochastic_rounding: bool = env NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING,
    disable_2d_quantization: bool = env NVTE_NVFP4_DISABLE_2D_QUANTIZATION,
    row_scaled_activation: bool = env NVTE_NVFP4_ROW_SCALED_ACTIVATION,
    nvfp4_4over6: str = "none",              # adaptive scaling; aimed at RL/post-training, not
    nvfp4_4over6_e4m3_use_256: str = "all",  # pretraining-with-RHT
    nvfp4_4over6_err_mode: str = "MAE",
    fp4_format: Format = Format.E2M1,
    fp8_format: Format = Format.E4M3,
    fp8_dpa: bool = False, fp8_mha: bool = False,
    backward_override: Optional[str] = None,
)
```

Note the polarity: **every stabiliser is on by default and the arguments only disable them.** Sibling
recipes: `MXFP8BlockScaling` (32 values, E8M0), `Float8BlockScaling`, `Float8CurrentScaling`,
`DelayedScaling`, and an experimental `CustomRecipe(qfactory=…, quantization_alignment=128)`.

Autocast: `fp8_autocast(enabled, calibrating, fp8_recipe, fp8_group, _graph)` is **deprecated** in
favour of `autocast(enabled, calibrating, recipe, amax_reduction_group, _graph)` in
`transformer_engine.pytorch.quantization`. The notebook probes for both.

Support gates, quoted from TE:

- `check_fp8_support()` — cc ≥ 9.0, or 8.9 (Ada) with cuBLASLt ≥ 12.1.3
- `check_mxfp8_support()` — cc ≥ 10.0, **and explicitly not supported on 12.0+**
- `check_nvfp4_support()` — cc ≥ 10.0: *"Device compute capability 10.0 or higher required for NVFP4
  execution."*
- TE docs: SM100 (Blackwell) or later for training and inference; RHT is BF16-input only

The MXFP8-excludes-sm120 / NVFP4-does-not asymmetry is the load-bearing detail for consumer Blackwell.

---

## 3. Hardware and software reality check

The development machine is an **RTX 5070** — consumer Blackwell, `sm_120`, 12 GB, on **Windows 11**.

- **FP4 silicon:** RTX 50-series are the first consumer GPUs with FP4 tensor cores; NVIDIA lists
  B200, B300, RTX 5090 and RTX PRO 6000 as native-NVFP4 parts. `sm_120` reports compute capability
  12.0, which satisfies TE's `>= 10.0` NVFP4 gate.
- **Toolchain:** build with `TORCH_CUDA_ARCH_LIST` including `12.0 12.1`; PyTorch ≥ 2.9, CUDA 12.8+/13.x.
  TensorRT-LLM's early `"FP4 Gemm not supported before Blackwell, nor GeForce Blackwell"` guard was
  relaxed to a plain `SM >= 100` check, i.e. GeForce Blackwell is no longer excluded by policy.
- **Measured training speedups (nanochat TE thread, RTX 5090 / RTX PRO 6000 / B200):** FP8 ~20–30%
  over baseline; NVFP4 adds ~20% on some configurations and is *slower* than FP8 on others. Same
  thread: logit soft-capping costs ~15% throughput — a direct argument for QK-norm instead.
- **Windows:** TE ships Linux wheels; use **WSL2**. The notebook therefore never assumes TE imports
  successfully. Run natively on Windows and `PRECISION_PATH` resolves to BF16 — the FP4 recipe, the
  A/B table and the ablation grid all degrade to documented no-ops, so native Windows is the
  *correctness* path and WSL2 is the *precision* path. A second native-Windows constraint is
  `torch.compile`: inductor needs Triton, which has no Windows backend, and the failure surfaces
  lazily inside the first forward pass rather than at the `torch.compile()` call — so `cfg.compile`
  is forced off on Windows in the config cell instead of being caught in the training cell.
  Verified natively on torch 2.7.1+cu128 / sm_120; the default PyPI `torch` wheel is CPU-only and
  must be replaced with a cu128+ build.
- **12 GB:** the `laptop` preset (1024×16, seq 1024) is a genuine run; the 13k-tok/param budget is
  not reachable there. This tension is stated in the notebook's first cell rather than hidden.

**Mapped to:** the capability-probe cell, `PRECISION_PATH` degrading NVFP4 → MXFP8 → FP8 → BF16, and
three presets sharing one config dataclass.

---

## 4. Tokenizer

### 4.1 SuperBPE — arXiv:2503.13423

Standard BPE cannot merge across the pretokenizer's whitespace boundaries, so a subword vocabulary can
never contain `def main(`, ` unsafe fn `, or `by the way`. SuperBPE adds a **pretokenization
curriculum**: learn subwords for the first `t` merges with pretokenization on, then learn
**superwords** with it off; encode with it off.

- up to **33% fewer tokens** than BPE at a fixed vocabulary size on general text
- an **8.1B** model: **+4.0 points average across 30 tasks**, **+8.2 on MMLU (44.7 vs 36.5)**, better
  on 25/30 tasks
- **−27% FLOPs per input byte**, proportional to the token reduction — training *and* inference

### 4.2 BoundlessBPE — arXiv:2504.00178

Reaches the same relaxation from the other direction: standard BPE has diminishing returns at larger
vocabularies because pretokens are already atomic; merging complete pretokens into superwords restores
scaling.

### 4.3 Cost, and the fast trainer — arXiv:2604.05192

The honest drawback: **BoundlessBPE took 4.7 CPU-days on 1 GB** where HF BPE takes **59 s**; SuperBPE
needed ~100 CPUs for a few hours on 10 GB. *Faster Superword Tokenization* (April 2026) restructures
this with a transition point and greedy single-pass matching, deferring expensive comparisons.

**Mapped to:** stage 1 uses the fast Rust `tokenizers` BPE trainer with a GPT-4-style split regex;
stage 2 is implemented directly as greedy pair-counting over the stage-1 **id stream** on a bounded
sample (`sample_bytes`, default 400 MB) — the same algorithm, with the cost made an explicit knob. The
result is written back as extra `merges` on a plain HF `BPE` model whose pre-tokenizer is
`ByteLevel(use_regex=False)`, so *inference with pretokenization off* is automatic and the artefact
loads in vanilla `transformers`.

### 4.4 Vocabulary size — 32768, and why not 64k

Three independent lines of evidence converge on a 32k-class vocabulary, which is why an earlier
draft's 65,536 was wrong:

1. **Measured locally.** `TOKENIZER-V3.md`'s sweep (§1): 32k `xw` reaches 3.4185 combined bytes/token
   against 64k `v2`'s 3.3532 — *better compression at half the vocabulary* — and 8232 vs 7354 bytes
   of corpus consumed per second, because the `[B, L, V]` logit matmul and the tied embedding both
   halve. Bytes/token alone would have chosen 64k; corpus throughput is the metric that decides.
2. **Predicted by theory.** *Scaling Laws with Vocabulary* (arXiv:2407.13623, NeurIPS 2024) fits
   compute-optimal vocabulary against non-vocabulary parameters and lands on 32–38k at ~300M — and
   states the optimum moves *lower* when data is the bottleneck, which is this run exactly.
3. **Constraints.** 32768 is divisible by 128 (FP4/FP8 GEMM and TMA tile alignment; TE's
   `CustomRecipe` exposes `quantization_alignment: int = 128`) and every id fits `uint16`, halving
   shard bytes. Byte-level coverage makes **UNK unrepresentable**, replacing SentencePiece's
   `byte_fallback`.

For contrast Gemma 4 uses 262k — right for a 31B multilingual multimodal model, and ~17% of this
model's parameters if copied blindly.

**The id space is wider than the vocabulary.** Stage 2 can mint two superword merges whose
byte-level strings coincide; the duplicates collapse in the vocabulary map, so `get_vocab_size()`
returns fewer pieces than `max(id) + 1` — a handful of ids exist but are unused. Sizing the
embedding by the *count* leaves the top ids unmapped, and the first id the tokenizer actually emits
above that bound indexes past the embedding table. CUDA reports this as a device-side assert raised
several kernels later inside the first backward pass (observed in an `RMSNorm` forward), which points
nowhere near the cause. The notebook therefore sizes the embedding by the id space, rounded up to a
multiple of 128 — which restores the 32768 alignment the vocabulary target asked for in the first
place.

The bottom 64 ids are reserved: chat surface (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|think|>`,
`<|tool_call|>`, `<|tool_result|>`, `<|eot|>`), Aletheia pipeline stages (`<|intent|>`, `<|context|>`,
`<|plan|>`, `<|capability|>`, `<|policy|>`, `<|action|>`, `<|provenance|>`, `<|memory|>`,
`<|entity|>`, `<|component|>`), source structure (`<|file|>`, `<|code|>`, `<|diff|>`, `<|adr|>`,
`<|doc|>`), and spare `<|reserved_N|>` slots so future tokens never renumber the vocabulary.

### 4.5 Fitting corpus — chat-first

The tokenizer is fitted to what this model will actually read: ~58% prose (Nemotron-CC-v2,
FineWeb-Edu), 22% code, 12% Aletheia OS (so `CapabilityRef`, `ContextFabric` and `IntentEnvelope` are
single tokens), 8% math. `TOKENIZER-V3.md`'s first isolated finding — fitting the tokenizer to the
*mixture* rather than to one half of it was free on the local half and +5.8% on the other — is the
reason this is specified as a mixture rather than "the repo".

---

## 5. Architecture

### 5.1 What the August-2026 dense reference stacks do

**Gemma 4 (arXiv:2607.02770, April 2026)** — E2B (2.3B effective), E4B (4.5B), 12B, 26B-A4B MoE, 31B:

- local:global attention **4:1 for E2B, 5:1 for the rest**
- **pp-RoPE with p=0.25 on global layers**, ordinary RoPE on local layers; **RoPE base 1M global /
  10k local**
- **pre-norm and post-norm RMSNorm**, plus **QK-norm** (Gemma 3 already replaced soft-capping with
  QK-norm)
- 262k vocabulary, bfloat16 training on TPUv5p/v6e, ZeRO-3, JAX + Pathways, data cutoff Jan 2025

**Qwen3.5 (Feb–Mar 2026)** — 0.8B → 9B sharing one architecture, Apache-2.0; QK-norm, 8 KV heads;
Qwen3.5-4B native 262,144 context extensible past 1M.

**MiniCPM5-1B / MiniCPM-SALA (Feb 2026)** — 25% sparse attention (InfLLM-v2) + 75% linear attention
(Lightning Attention), 3.5× inference speed at million-token context. Notable because Aletheia OS's
AI subsystem already pins MiniCPM GGUF references.

**LFM2.5-2.6B (Liquid AI, Aug 2026)** — 128k context, tool calling, runs on a Raspberry Pi; the
token-budget reference in §7.

### 5.2 Hybrid attention

- *Rethinking the Role of Efficient Attention in Hybrid Architectures* — arXiv:2606.15378
- *FlashMorph: Fast LAyer Selection for Hybrid MORPHing* — arXiv:2606.30562: conversion quality
  **critically depends on which layers keep full attention**; heuristic fixed-placement and
  layerwise-scoring selection both ignore interdependent layer effects
- *Hymba* — hybrid attention/SSM heads *within* a layer, plus learnable meta tokens
- *Nemotron-Flash* (arXiv:2511.18890), *Nemotron-H* (arXiv:2504.03624), *Nemotron 3 Ultra*
  (arXiv:2606.15007) — hybrid Mamba-Transformer at scale; the NVFP4 paper's own 12B is one of these
- the widely adopted practice: keep a small fraction of full-attention layers among efficient ones

**Decision.** Hybrid **is** the default: 3 efficient layers per full-attention layer, with the
efficient slot filled by **gated DeltaNet** when `flash-linear-attention` is present and by
**sliding-window attention (1024)** otherwise. FlashMorph's finding is about *placement*, not about
whether to hybridise — so the full-attention layers are evenly spaced, which is the one placement
every study agrees on, and the fallback path keeps every mixer matmul inside `te.Linear` where NVFP4
applies. `cfg.hybrid_linear=False` reverts to pure SWA:global for ablation.

### 5.3 Final configuration and why each element is there

Default preset is `rtx5070`: 1024 × 24, `d_ff=2816`, 16 heads / 4 KV × 64, ~305M params, sized so a
12 GB card can actually finish a run. `workstation` and `cluster` scale width, depth and MoE.

| Element | Value | Source / reason |
|---|---|---|
| depth × width | 24 × 1024, `d_ff=2816` (~8/3·d) | all dims %128 for FP4 alignment; fits 12 GB with FP32 masters + Muon momentum + activations |
| GQA | 16 heads / 4 KV, `head_dim=64` | KV cache is the on-device constraint; 4:1 is the Qwen/Llama consensus |
| hybrid attention | 3 gated-DeltaNet-or-SWA(1024) : 1 full | §5.2 |
| attention sinks | learnable per-head key with a zero value | StreamingLLM / GPT-OSS. Implemented as a prepended K/V pair so it composes with SDPA instead of needing a hand-written softmax |
| QK-norm | RMSNorm on q and k | Gemma 3/4; bounds attention logits, which matters more in 4-bit; avoids soft-capping's ~15% throughput cost |
| pp-RoPE | rotate 25% of head dims on global layers; θ 10k local / 1M global | Gemma 4's `p=0.25` split-base scheme. Mellum2's single θ=500k is the alternative for a code-only model |
| norms | pre + post RMSNorm per sublayer | Gemma 4; stabilises the residual stream that FP4 rounding perturbs |
| MoE | top-2 of 8 routed + 1 shared expert, aux-loss-free bias balancing | DeepSeek-V3 bias nudging; Qwen3 / Gemma 4 26B-A4B shared-expert shape. Off at 12 GB — MoE buys FLOPs with memory, and memory is what is missing |
| MTP | 1 extra block predicting `t+2`, weight 0.3 | DeepSeek-V3. Data efficiency, plus free speculative decoding |
| intra-doc masking | no attention across `<\|eos\|>` inside a packed sequence | packing without it teaches cross-document dependence that does not exist |
| activation / bias / tying | SwiGLU, no biases, tied embeddings | universal in 2026 dense stacks |
| logits | FP32 + **z-loss 1e-4** | keeps `logsumexp` bounded; also removes the predecessor's chunked-logit workaround |
| init | trunc-normal `d_model^-0.5`, residual projections ÷ `sqrt(2·n_layers)` | GPT-NeoX/OLMo depth scaling |
| master weights | FP32, compute in BF16/FP4 | repairs the bf16-master defect measured in `aletheia-lm/config.py` (§1) |
| EMA | optional (`ema_decay`) | cheap final-checkpoint gain; costs one FP32 copy |

**Declined deliberately**, because "everything SOTA" also means not stacking components that fight:

* **Full sparse attention (InfLLM-v2 / MiniCPM-SALA)** — its win appears at ≥128k context; this model
  trains at 2048 and extends in the anneal.
* **Mamba-2 / SSM blocks** — gated DeltaNet already occupies the efficient slot, and TE's FP4 path
  covers `Linear`, not scan kernels.
* **Hyper-connections / value-residual variants** — each changes the residual scale, invalidating the
  depth-scaled init the 4-bit run depends on.
* **MLA (latent KV)** — a genuine KV win, but its interaction with partial RoPE is unpublished at
  this scale.
* **Distillation** — the strongest single quality lever at ~300M, and explicitly out of scope: this
  model is not derived from any other model.

---

## 6. Optimizer and schedule

### 6.1 Muon

- *Practical Efficiency of Muon for Pretraining* — arXiv:2505.02222: Muon **expands the Pareto
  frontier over AdamW** on the compute-time trade-off and **retains data efficiency at batch sizes far
  beyond the critical batch size**.
- *SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales* — arXiv:2607.20548 (NVIDIA, July 2026):
  with **update-RMS matching** for fair LR transfer, **SOAP and Muon consistently beat AdamW** on
  multi-billion-parameter models over trillions of tokens. Identifies and fixes SOAP instabilities at
  large batch (per-step QR orthogonalization).
- *Benchmarking Optimizers for LLM Pretraining* — arXiv:2509.01440 / *Fantastic Pretraining
  Optimizers* — arXiv:2509.02046: **Muon is best in smaller data-to-model ratio regimes but is
  outperformed by Kron and SOAP once the ratio reaches 8× Chinchilla or more.** AdamW degrades past
  the critical batch size; Muon and SOAP stay token-efficient to ~100M-token global batches.
- Adoption: Kimi K2, GLM-4.5, INTELLECT-3 (2025); `torch.optim.Muon` ships in PyTorch 2.9 —
  `lr=1e-3`, `weight_decay=0.1`, `momentum=0.95`, `nesterov=True`, `ns_steps=5`, **2D parameters
  only**, with AdamW for biases/embeddings.

**Caveat recorded, not hidden.** This project runs at ~650× Chinchilla, exactly the regime where the
benchmark papers say Kron/SOAP overtake Muon. Muon is chosen anyway for the combination of large-batch
tolerance, a native PyTorch implementation, and production evidence; SOAP is the first thing to try if
the loss curve plateaus. **Mapped to:** `split_params` routing 2D non-embedding matrices to Muon and
everything else to AdamW, with a bundled Newton–Schulz Muon (including Jordan's
`0.2·sqrt(max(dim))` RMS match) for torch < 2.9.

### 6.2 WSD schedule

Warmup → constant → `1−sqrt` decay, with `decay_frac=0.20`. Two properties the predecessor's cosine
lacks: the total step count need not be fixed in advance (a run can be extended without corrupting the
schedule), and the decay phase is a natural place to co-locate other transitions. The notebook
deliberately aligns `bf16_switch_frac=0.82` with the start of decay, so the final 18% of tokens are
simultaneously high-precision and low-LR — which is where arXiv:2509.25149 recovers most of the
1.5% → 0.5% loss gap.

---

## 7. Token budget

**LFM2.5-2.6B (Liquid AI, August 2026): ~34T tokens on 2.69B parameters ≈ 12,600–13,000 tokens per
parameter.** Chinchilla-optimal is ~20. Context: Llama-2-7B ~290× the Chinchilla ratio, Gemma-7B
~857×, Gemma-2-9B ~889×. Overtraining is not an exotic claim in 2026, it is the default for anything
intended to be deployed, because inference cost scales with parameter count and quality keeps
improving well past the compute-optimal point. *Test-Time Scaling Makes Overtraining Compute-Optimal*
(arXiv:2604.01411) makes the formal version of the argument.

**Mapped to, honestly.** `tokens_per_param = 13_000` is kept as the *aspiration*, and the config cell
prints what it would actually cost: at ~271M non-embedding parameters that is ~3.5T tokens, i.e.
GPU-years on an RTX 5070. So the run is sized by **wall clock** (`train_days`, default 14) and the
cell reports the tok/param that budget really buys, next to the 13,000 target. The ratio is reachable
on the `cluster` preset and is not reachable here; the notebook says which one you are in.

Two further constraints that any 13k-tok/param plan has to satisfy and that are easy to miss:

* **It is a data problem before it is a compute problem.** LFM2.5 spent ~34T *unique* tokens. Looping
  a 17B-token corpus 200× instead does not get you there: repetition decay (arXiv:2305.16264,
  R\*_D ≈ 15) says the returns collapse long before that. Reaching 13k tok/param means streaming
  Nemotron-CC/FineWeb-Edu/DCLM at trillion-token scale, which is why the data cell streams the Hub
  rather than reading a local corpus.
* **Domain knowledge is the exception.** The Aletheia OS repo is a few megabytes, so `aletheia_epochs
  = 24` is deliberate repetition of a tiny, high-value slice — a different regime from the bulk mix,
  and the one place where re-reading is the point.

---

## 8. Data

- **Nemotron-CC / Nemotron-CC-v2** (arXiv:2412.02595): classifier ensembling + synthetic rephrasing +
  fewer heuristic filters. A high-quality subset improves **MMLU by +5.6 over DCLM** at 8B/1T; the
  full 6.3T set matches DCLM on MMLU while holding **4× more unique real tokens**. It is gated, so it
  is the *first* candidate of the `web_hq` slot rather than its only one.
- **FineWeb-Edu** and **DCLM**: both gain from aggressive model-based filtering but discard ~90% of
  data. Their union de-correlates filter bias — the **Zyda-2** construction (arXiv:2411.15242) is
  5T tokens of FineWeb-Edu3 + DCLM + Zyda-1 + Dolma-CC with cross-deduplication. **Zyda-2 is the
  ungated `web_hq` default**, which is a strict improvement on the previous design: that spent three
  separate slots (`nemotron_cc_hq` / `fineweb_edu` / `dclm`) re-assembling this union by hand while
  double-counting the overlap Zyda-2 already removes.
- **Cosmopedia-v2** (`HuggingFaceTB/smollm-corpus`): 28B tokens of textbook-style synthetic rewrites,
  the open instance of the synthetic-rephrasing axis Nemotron-CC-HQ's MMLU gain is credited to. Its
  own `synthetic_edu` slot, because rephrased and crawled text are different distributions and the
  mixture should be able to re-weight them independently.
- **Dolmino-mix-1124**: OLMo 2's mid-training mix (quality-weighted web + reference + STEM) as the
  `curated` slot.
- **Ultra-FineWeb**: two-stage annealing as a cheap data-quality verifier — the pattern behind
  `cfg.mix` being a set of *fractions of budget* that can be re-weighted from a short anneal probe.
- **Code**: `bigcode/starcoderdata` (all language configs interleaved) when the grant is held,
  otherwise `OpenCoder-LLM/opc-annealing-corpus` across its three configs. The target model reads and
  writes Rust, C and assembly about a kernel. **The Stack v2 is not usable as a text stream**: its
  rows carry Software Heritage blob ids, and the contents require a separate S3 fetch with AWS
  credentials — a `row["content"]` pipeline reads it as an unbounded run of empty rows rather than an
  error. `starcoder2data-extras` is a different trap: its configs are arxiv / issues / wikipedia, so
  it streams successfully while contributing no code.
- **Math**: `HuggingFaceTB/finemath` `finemath-4plus`, the higher-precision successor to OpenWebMath
  on the same crawl; OpenWebMath remains the fallback.
- **Access and degradation**: the mixture is built entirely from **ungated** corpora. The gated sets
  (Nemotron-CC-v2, starcoderdata) sit first in their chains so an account holding the grant uses them
  automatically. Each slot is an ordered *candidate chain* — first source that authenticates and
  actually yields text wins — and the shard builder closes with a realised-vs-requested mixture table
  so neither a missing grant nor an exhausted stream can skew the blend behind a one-line warning.
  Nothing is downloaded: every corpus is consumed with `streaming=True` and only `uint16` shards land
  on disk.
- **SmolLM3 / SmolTalk2** recipe: mid-training as a distinct phase — SmolLM3 used 35B tokens of
  OpenThoughts3-1.2M plus Llama-Nemotron-Post-Training v1.1 (R1 traces) for ~140B tokens over 4
  epochs, then SFT. SmolTalk2 is ~3.4M samples (OpenThoughts, Tulu 3, multilingual), split
  Mid / SFT / Preference, decontaminated against benchmarks. The `SFT` config has **no `train`
  split** — it is 25 named sub-corpora, and interleaving all of them is the recipe; requesting
  `train` raises, which would silently zero the chat share of the mixture.
- **Aletheia OS repo**: upsampled `aletheia_epochs` times, plus two mechanically generated sets —
  heading-anchored doc QA and Rust-item QA (each answer quotes real file content and cites the path),
  and templated `intent → context → plan → capability → policy → action → provenance` traces so the
  OS's control flow is learned as a conversational form. No teacher model is involved, so the facts
  are the repository's facts.

**Mapped to:** `cfg.mix = {web_hq .34, synthetic_edu .18, curated .10, chat .14, code .11, math .05,
aletheia .08}` — slot names are mixture *roles*, each resolved through a candidate chain — plus
streaming ingestion and a masked anneal phase (loss on assistant spans only) co-located with the
BF16 tail.

---

## 9. Evaluation

**lm-evaluation-harness** (EleutherAI) is the standard: 60+ benchmarks, 300+ tasks. The Open LLM
Leaderboard **v2** replaced the saturated v1 set with **MMLU-Pro, GPQA, MuSR, IFEval, BBH,
MATH Level 5**.

Stated honestly in the notebook: at ~1.5B with a modest token budget, the hard suite sits near chance
and is close to uninformative. The signals that actually move at this scale are held-out perplexity
(reported **per domain**), the easy suite (HellaSwag, ARC, PIQA, WinoGrande, OpenBookQA,
LAMBADA, MMLU), and the domain probe.

**No public benchmark measures whether a model knows what a Context Fabric is**, so §11a is a
closed-book, keyword-scored probe over ten facts drawn from the Aletheia repository: what Aletheia is,
kernel language, target architectures, the seven domain primitives, the three capability properties,
policy-vs-capability independence, Context-Fabric-not-RAG, WASM isolation with no ambient authority,
the full deterministic pipeline, and the untrusted status of the AI subsystem.

Beyond quality, the notebook measures the three axes that decide whether NVFP4 was worth it:
**relative validation-loss gap vs a BF16 twin** (bar: <1.5% mid-training, ~0.5% after the tail),
**tokens/s and MFU per recipe**, and **peak memory per recipe** — plus an ablation grid over exactly
the knobs the papers disagree about (RHT on/off, stochastic rounding on/off, 2D weight scaling
on/off, BF16 block count, FP4 attention, deterministic vs random Hadamard, Muon vs AdamW, SuperBPE vs
stage-1 BPE).

---

## 10. Open risks

1. **`sm_120` NVFP4 *training* is inferred, not published.** `check_nvfp4_support()` gates on
   cc ≥ 10.0 and 12.0 passes; RTX 5090 numbers exist in the nanochat TE thread. But no vendor
   document promises NVFP4 *training* parity on GeForce Blackwell, and MXFP8 is explicitly excluded on
   12.0+. First run should be the §9a A/B, not a full budget.
2. **Speedup expectations.** FP4 is 2–3× the *arithmetic* of FP8; measured end-to-end is ~9–10%
   (arXiv:2605.09825) to ~20% (nanochat), and sometimes negative. Non-GEMM work does not shrink.
3. **Wgrad instability** may appear only after FP4 reaches gradients. Remedy is
   `deterministic_hadamard`, not LR surgery.
4. **Optimizer choice at 650× Chinchilla** favours Kron/SOAP over Muon per arXiv:2509.01440. Muon is
   the pragmatic pick, not the benchmark winner in this exact regime.
5. **Stage-2 tokenizer cost** is quadratic-ish in merge count over the sampled stream. `sample_bytes`
   bounds it; a smaller sample yields slightly worse superwords.
6. **`NVTE_NVFP4_DETERMINISTIC_RHT`** is set by the notebook when `deterministic_hadamard=True`, but
   the env-var name is not in the TE recipe dataclass this dossier verified — treat it as
   build-dependent and confirm against your installed TE before relying on it.
7. **Distillation is absent.** It is the strongest single quality lever at 1.5B and is orthogonal to
   everything here; it needs a teacher and a licence decision.
8. **The `laptop` preset is a smoke test.** 12 GB and 13k tok/param do not coexist.

---

## 11. Source index

Primary — 4-bit training:

- Pretraining Large Language Models with NVFP4 — https://arxiv.org/abs/2509.25149
- Pretraining LLMs with MXFP4 on Native FP4 Hardware — https://arxiv.org/abs/2605.09825
- Full-Stack FP4 — https://arxiv.org/pdf/2607.04422
- TE NVFP4 feature docs — https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html
- TE recipe source — https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe/__init__.py
- TE quantization/autocast source — https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py
- TE FP8/FP4 primer — https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
- NVIDIA 4-bit methodology writeup — https://www.marktechpost.com/2026/05/18/nvidia-introduces-a-4-bit-pretraining-methodology-using-nvfp4-validated-on-a-12b-hybrid-mamba-transformer-at-10t-token-horizon/

Hardware:

- FP8/NVFP4 training with TE, nanochat discussion #382 — https://github.com/karpathy/nanochat/discussions/382
- TensorRT-LLM issue #5018, 5090 NVFP4 — https://github.com/NVIDIA/TensorRT-LLM/issues/5018
- Unlocking NVFP4 speed on consumer GPUs — https://dasroot.net/posts/2026/04/unlocking-nvfp4-speed-consumer-gpus/
- TensorRT FP4 on GeForce RTX 50 — https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/

Tokenizer:

- SuperBPE: Space Travel for Language Models — https://arxiv.org/pdf/2503.13423
- Boundless Byte Pair Encoding — https://arxiv.org/pdf/2504.00178
- Faster Superword Tokenization — https://arxiv.org/pdf/2604.05192
- Getting the most out of your tokenizer — https://arxiv.org/html/2402.01035v2
- Scaling Laws with Vocabulary — https://arxiv.org/abs/2407.13623

Architecture:

- Gemma 4 Technical Report — https://arxiv.org/pdf/2607.02770
- Gemma 3 Technical Report — https://arxiv.org/pdf/2503.19786
- Rethinking Efficient Attention in Hybrid Architectures — https://arxiv.org/abs/2606.15378
- FlashMorph — https://arxiv.org/abs/2606.30562
- Hymba — https://arxiv.org/html/2411.13676
- Nemotron-Flash — https://arxiv.org/pdf/2511.18890
- Nemotron-H — https://arxiv.org/pdf/2504.03624
- Nemotron 3 Ultra — https://arxiv.org/pdf/2606.15007
- Mellum2 (3:1 SWA, theta=500k, Muon for a code model) — https://arxiv.org/abs/2605.31268
- MiniCPM / MiniCPM-SALA — https://github.com/openbmb/minicpm
- LFM2 Technical Report — https://arxiv.org/html/2511.23404v1
- LFM2.5-2.6B — https://www.liquid.ai/blog/lfm2-5-2-6b

Optimizer:

- Practical Efficiency of Muon for Pretraining — https://arxiv.org/html/2505.02222v1
- SOAP, Muon, and Beyond — https://arxiv.org/abs/2607.20548
- Benchmarking Optimizers for LLM Pretraining — https://arxiv.org/abs/2509.01440
- Fantastic Pretraining Optimizers and Where to Find Them — https://arxiv.org/html/2509.02046v2
- torch.optim.Muon — https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py

Data and budget:

- Nemotron-CC — https://arxiv.org/abs/2412.02595
- Zamba2 / Zyda-2 — https://arxiv.org/pdf/2411.15242
- SmolLM3 — https://huggingface.co/blog/smollm3
- SmolTalk2 — https://huggingface.co/datasets/HuggingFaceTB/smoltalk2
- SmolLM2 — https://arxiv.org/pdf/2502.02737
- Test-Time Scaling Makes Overtraining Compute-Optimal — https://arxiv.org/pdf/2604.01411
- LFM2.5-2.6B release coverage — https://www.marktechpost.com/2026/08/06/liquid-ai-lfm2-5-2-6b-on-device-agentic-model/

Evaluation:

- lm-evaluation-harness task list — https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md

Baseline repositories:

- Aletheia OS — https://github.com/hotocoo/aletheia
- aletheia-lm (private; the code model, source of the tokenizer sweep in §1) — https://github.com/hotocoo/aletheia-lm
- aletheia1Bmx (MLX predecessor) — https://github.com/hotocoo/aletheia1Bmx
- aletheiatokenizer (64k unigram predecessor) — https://github.com/hotocoo/aletheiatokenizer
