# Aletheia-NVFP4

4-bit (NVFP4) pretraining of **Aletheia-Chat** — a conversational model whose domain knowledge is
[Aletheia OS](https://github.com/hotocoo/aletheia): its microkernel, capability engine, policy engine,
Context Fabric, WASM component runtime and ADRs.

Successor to [`aletheia1Bmx`](https://github.com/hotocoo/aletheia1Bmx) (MLX / bfloat16 / Apple
Silicon) and to [`aletheiatokenizer`](https://github.com/hotocoo/aletheiatokenizer) (64k
code-only SentencePiece unigram).

| | |
|---|---|
| **Notebook** | [`Aletheia_NVFP4_Pretrain.ipynb`](Aletheia_NVFP4_Pretrain.ipynb) — 39 cells, end to end |
| **Research** | [`RESEARCH.md`](RESEARCH.md) — every decision, its August-2026 source, and the disagreements between sources |
| **Generator** | [`tools/build_notebook.py`](tools/build_notebook.py) — the notebook is generated from this, so it stays diffable |

## What it does, in order

1. **Capability probe** — picks NVFP4 → MXFP8 → FP8 → BF16 based on what the GPU and Transformer
   Engine actually support, and says which path it took.
2. **Tokenizer** — trains a **SuperBPE-style superword tokenizer** (`vocab=65536`, byte-level, no
   UNK) on Aletheia OS + prose + code, with chat and OS-pipeline special tokens; benchmarks
   bytes/token against plain BPE, the old `aletheiacode64k`, and `cl100k`/`o200k`.
3. **Data** — streams Nemotron-CC-v2 / FineWeb-Edu / DCLM / the-stack-v2 / open-web-math / SmolTalk2
   into `uint16` memmap shards with a document index, and upsamples the Aletheia repo ~24×.
4. **Model** — dense decoder, GQA + QK-norm, pre+post RMSNorm, SwiGLU, sliding-window : global = 3:1,
   split RoPE bases (10k local / 1M global), tied embeddings, FP32 logits + z-loss. Optional gated
   DeltaNet hybrid.
5. **Training** — `NVFP4BlockScaling` (16×16 RHT on Wgrad, stochastic rounding on gradients, 2D weight
   scaling), first 2 + last 4 blocks in BF16, **NVFP4 → BF16 for the last 18% of tokens**, Muon on 2D
   matrices + AdamW elsewhere, WSD schedule, grad accumulation, activation checkpointing,
   `torch.compile`, resumable checkpoints.
6. **Benchmarks** — recipe A/B for relative loss gap, tokens/s, MFU and peak memory; KV-cache and
   decode latency; an ablation grid over every contested knob.
7. **Anneal** — masked chat mid-training on SmolTalk2 + mechanically generated Aletheia QA + templated
   `intent → context → plan → capability → policy → action → provenance` traces.
8. **Evaluation** — lm-eval-harness v2 hard suite and easy suite invocations, per-domain perplexity,
   and a closed-book **Aletheia knowledge probe** over ten facts from the repository.
9. **Chat** — KV-cached generation behind the chat template.

## Requirements

Linux or **WSL2** (Transformer Engine ships Linux wheels only), CUDA 12.8+/13.x, an NVIDIA GPU with
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
`fp8_autocast`, and to a bundled Newton–Schulz Muon when `torch.optim.Muon` is absent. It was smoke
tested end-to-end (tokenizer → shards → model → train → anneal → probe) on torch 2.7.1 CPU /
transformers 4.55 / tokenizers 0.21.

## Scale

`PRESET` in the config cell selects one of three coherent shapes. Nothing else needs changing.

| preset | shape | token budget | target |
|---|---|---|---|
| `laptop` | 1024 × 16, seq 1024 | 400 tok/param | 12 GB RTX 5070 smoke run |
| `workstation` | 2048 × 30, seq 2048, mb 4×16 | 13,000 tok/param | RTX PRO 6000 / H200 |
| `cluster` | 2048 × 30, seq 4096, mb 8×24 | 13,000 tok/param | multi-node B200 |

**Chinchilla is deliberately ignored.** The budget is ~13,000 tokens per non-embedding parameter — the
LFM2.5-2.6B ratio (~34T tokens / 2.69B params), roughly 650× compute-optimal. Reasoning and sources in
[`RESEARCH.md §7`](RESEARCH.md#7-token-budget).

## Honest limitations

- NVFP4 *training* on consumer `sm_120` is inferred from TE's `cc >= 10.0` gate and third-party
  RTX 5090 measurements, not from a vendor guarantee. Run the recipe A/B before spending a budget.
- Measured FP4 end-to-end speedup is ~9–20% over FP8, not the 2–3× arithmetic ratio: non-GEMM work
  does not shrink.
- A 12 GB GPU cannot reach 13k tok/param in human time. The `laptop` preset is a correctness test.
- No distillation. It is the strongest quality lever at this scale and is out of scope here.

Full risk list: [`RESEARCH.md §10`](RESEARCH.md#10-open-risks).

## License

MIT.
