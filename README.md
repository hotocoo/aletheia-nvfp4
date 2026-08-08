# Aletheia-NVFP4

A **new** chat model, pretrained from scratch in **NVFP4** (4-bit) on one consumer Blackwell GPU.
Nothing is inherited: its own tokenizer, its own architecture, random init, no distillation, no
weights from any other model. Its domain knowledge is
[Aletheia OS](https://github.com/hotocoo/aletheia) — kernel, capability engine, policy engine,
Context Fabric, WASM component runtime, ADRs.

Not to be confused with `aletheia-lm`, which is a **code** model. This one talks.

| | |
|---|---|
| **Notebook** | [`Aletheia_NVFP4_Pretrain.ipynb`](Aletheia_NVFP4_Pretrain.ipynb) — tokenizer → data → model → 4-bit training → benchmarks → chat anneal → eval → chat |
| **Research** | [`RESEARCH.md`](RESEARCH.md) — every decision, its August-2026 source, and where sources disagree |
| **Generator** | [`tools/build_notebook.py`](tools/build_notebook.py) — the notebook is generated from this, so it stays diffable |

## The stack (all August 2026, all on by default)

**Number format — NVFP4.** E2M1 values, FP8-E4M3 scale per 16, FP32 global scale. 16×16 random
Hadamard transform on Wgrad inputs, stochastic rounding on gradients only, 2D 16×16 weight scaling,
first 2 + last 4 blocks kept BF16, and a switch to BF16 for the final 18% of tokens. That is
arXiv:2509.25149's recipe, which held a 12B model to <1.5% relative loss versus FP8 over 10T tokens.

**Tokenizer — custom, built by the notebook.** A true two-stage **SuperBPE** superword tokenizer:
byte-level (no UNK), `vocab=32768`, stage 1 under a whitespace-crossing (`xw`) split regex, stage 2
with the barrier lifted so tokens may span spaces. The 32k-class vocabulary and the `xw` regex are
not guesses — they are the winning arm of a five-way sweep measured locally
([RESEARCH.md §1](RESEARCH.md#1-starting-point)), where `32k xw` beat `64k v2` outright. Stage 2 is
learned over the stage-1 id stream with an incremental linked-list + lazy-heap trainer, which is what
makes a faithful SuperBPE run possible without forking the Rust `BpeTrainer`.

**Architecture — every compatible SOTA component:**

- hybrid attention 3:1 — gated DeltaNet (or sliding-window 1024) : full attention
- GQA 4:1, QK-norm, learnable **attention sinks**, Gemma 4 **pp-RoPE** (25% rotation, θ 10k/1M split)
- pre+post RMSNorm, SwiGLU, no biases, tied embeddings, depth-scaled init
- **MoE** FFN — top-2 of 8 + 1 shared expert, aux-loss-free bias balancing (off on 12 GB, on above)
- **multi-token prediction** head (also gives speculative decoding for free)
- **intra-document masking** inside packed sequences, z-loss, FP32 logits
- **Muon** (2D matrices) + AdamW, **FP32 master weights**, warmup-hold-decay

Declined on purpose, with reasons, in [RESEARCH.md §5](RESEARCH.md#5-architecture).

## Requirements

Linux or **WSL2** (Transformer Engine ships Linux wheels only), CUDA 13.x, an NVIDIA GPU with
compute capability ≥ 10.0 for the FP4 path — B200/B300 (`sm_100`), RTX 5070/5080/5090 or
RTX PRO 6000 (`sm_120`). Without one, everything still runs in BF16.

Latest releases as of 2026-08-08 — torch **2.13.0**, transformer_engine **2.17.1**, transformers
**5.14.1**, tokenizers **0.23.1**, datasets **5.0.1**:

```bash
pip install --upgrade "torch>=2.13.0" --index-url https://download.pytorch.org/whl/cu130
pip install --no-build-isolation "transformer_engine[pytorch]>=2.17.1"   # Linux / WSL2 only
pip install -r requirements.txt
python tools/build_notebook.py        # regenerate the notebook after editing the generator
```

CUDA 13.0 is the PyPI default; CUDA 13.2 (`.../whl/nightly/cu132`) carries the widest Blackwell
support. For source builds on RTX 50-series, include `12.0 12.1` in `TORCH_CUDA_ARCH_LIST`.

Older stacks work too — the notebook falls back from TE's `autocast` to the deprecated
`fp8_autocast`, and to a bundled Newton–Schulz Muon when `torch.optim.Muon` is absent.

## How to pretrain it

```bash
wsl                                   # Transformer Engine needs Linux
jupyter lab Aletheia_NVFP4_Pretrain.ipynb
```

Then run the cells in order. What each stage costs on an RTX 5070:

| # | cell | what happens | time |
|---|---|---|---|
| 1 | probe | prints your precision path (NVFP4 / MXFP8 / FP8 / BF16) | seconds |
| 2 | config | `PRESET = "rtx5070"`; prints params, steps, and an honest budget reality check | seconds |
| 3 | tokenizer | clones Aletheia OS, streams ~1.5 GB of chat/prose/code, trains both SuperBPE stages, benchmarks bytes/token | ~1–2 h |
| 4 | data | streams the mixture into `uint16` shards, Aletheia upsampled 24× | hours, resumable |
| 5–7 | model / recipe / optimizer | builds the model, picks the recipe, splits Muon vs AdamW | seconds |
| 8 | **train** | the run. Checkpoints every 200 steps, resumes from `ckpt/manifest.json` | `train_days` (default 14) |
| 9 | benchmarks | NVFP4-vs-BF16 loss gap, tok/s, MFU, peak memory, KV cache, ablations | ~20 min |
| 10 | anneal | masked chat mid-training: SmolTalk2 + synthetic Aletheia QA + pipeline traces | ~1 h |
| 11 | eval | lm-eval invocations, per-domain perplexity, Aletheia knowledge probe | ~20 min |
| 12 | chat | talk to it | — |

Interrupting is safe at any point — the training cell resumes from the last checkpoint. To make the
run shorter or longer, change `train_days`; to change the model's shape, change `PRESET`.

## Scale

| preset | shape | target |
|---|---|---|
| `rtx5070` (default) | 1024 × 24, seq 2048, mb 2 × accum 8, dense | 12 GB consumer Blackwell |
| `workstation` | 1536 × 28, MoE 8×top-2, mb 8 × accum 4 | RTX PRO 6000 / H200 |
| `cluster` | 2048 × 32, MoE 32×top-2, seq 4096 | multi-node B200 |

**Chinchilla is deliberately ignored** — but so is arithmetic that doesn't hold. 13,000
tokens/parameter (the LFM2.5-2.6B ratio, ~34T tokens on 2.69B params) would need ~3.5T tokens for
this model, which is GPU-years on a 5070. The config cell prints exactly that comparison and then
sizes the run by wall clock instead of by wishful ratio. On a `cluster` preset the ratio is reachable;
on this machine it is not, and the notebook says so rather than quietly training for a week and
calling it SOTA.

## Honest limitations

- NVFP4 *training* on consumer `sm_120` is inferred from TE's `cc >= 10.0` gate plus third-party
  RTX 5090 measurements, not a vendor guarantee. Run the §9 recipe A/B before spending a budget.
- Measured FP4 end-to-end speedup is ~9–20% over FP8, not the 2–3× arithmetic ratio.
- Verified end to end only at toy scale in BF16 on CPU (all 23 code cells). The FP4 path has not been
  executed on this machine — it has no Transformer Engine and no Linux.
- No distillation. Strongest quality lever at this scale, out of scope for a from-scratch pretrain.

Full risk list: [`RESEARCH.md §10`](RESEARCH.md#10-open-risks).

## License

MIT.
