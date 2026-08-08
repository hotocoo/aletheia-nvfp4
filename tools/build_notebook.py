"""Build Aletheia_NVFP4_Pretrain.ipynb.

The notebook is generated from this file so that the Python source stays
readable and diffable. Run:  python tools/build_notebook.py
"""

import json
from pathlib import Path

cells = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


# ---------------------------------------------------------------- 0. title
md(r"""
# Aletheia-NVFP4 — 4-bit pretraining of an Aletheia OS assistant

**What this notebook is.** A **new model**, trained from scratch in 4-bit. Not a port, not a
continuation, not a distillation, not initialised from any existing checkpoint: `Aletheia-NVFP4` is
its own architecture with its own tokenizer, and the only thing it inherits from anywhere is
*evidence*. Its purpose is conversation grounded in *Aletheia OS*
(https://github.com/hotocoo/aletheia) — the kernel, capability model, policy engine, Context Fabric,
WASM component runtime and ADRs.

Where prior local measurement exists, it is used as evidence and cited as such — notably
`aletheia-lm/docs/TOKENIZER-V3.md`, whose five-arm sweep is why this notebook trains a **32k-class**
vocabulary with **whitespace-crossing merges** instead of the 64k/65k a naive reading of "bigger
vocab" would pick. Every such number below is a measurement someone ran, not an assumption. The
model itself starts at random init on cell one.

**Design date: August 2026.** Every major choice below is pinned to a primary source, and each is
re-checked at runtime by a capability probe rather than assumed.

| Layer | Choice | Why (source) |
|---|---|---|
| Number format | **NVFP4** (E2M1 + 1×16 E4M3 block scale + FP32 global scale), 2-level block scaling | *Pretraining LLMs with NVFP4*, arXiv:2509.25149 — 12B hybrid Mamba-Transformer, 10T tokens, MMLU-Pro 62.58% vs 62.62% FP8 |
| FP4 stability | 16×16 **Random Hadamard Transform on Wgrad inputs only**, **stochastic rounding on gradients only**, **2D 16×16 weight scaling**, RNE for weights/activations | ibid., §recipe; implemented as the defaults of `NVFP4BlockScaling` in Transformer Engine |
| Precision policy | first 2 blocks + last N blocks BF16 (~16–20% of linear layers); embeddings, LM head, norms, attention, optimizer state never FP4 | ibid. |
| Endgame | **switch NVFP4 → BF16 for the last ~18% of tokens** (relative loss gap 1.5% → 0.5%) | ibid. |
| Wgrad caution | AMD/PSU MXFP4 study: Wgrad quantization is the dominant divergence driver; *deterministic* Hadamard rotation stabilises it | arXiv:2605.09825 |
| Full-stack FP4 | optional FP4 optimizer state / attention path (kept **off** by default) | arXiv:2607.04422 |
| Tokenizer | **`aletheia_tok_v3`** — 32k byte-level BPE, whitespace-crossing (`xw`) split regex, 32 reserved special slots, `uint16` shards | measured in `aletheia-lm/docs/TOKENIZER-V3.md`: 32k `xw` beats 64k `v2` outright (3.4185 vs 3.3532 comb@0.9) and covers the corpus 3.1% faster; vocab-scaling theory agrees (arXiv:2407.13623 → 32–38k at ~300M non-embedding) |
| Tokenizer, optional | **true SuperBPE stage 2** on top of v3 — the experiment `TOKENIZER-V3.md` names as untried because `tokenizers` has no `BpeTrainer` warm-start | SuperBPE arXiv:2503.13423 (+4.0 pts avg / 30 tasks, −27% FLOPs/byte at 8B); BoundlessBPE arXiv:2504.00178; fast trainer arXiv:2604.05192. Implemented here without forking the Rust trainer — see §3 |
| Architecture | `aletheia-lm` `ModelConfig` unchanged: 1024×22, GQA 16/4, head_dim 64, SwiGLU 2816, **QK-norm**, tied embeddings, z-loss 1e-4, **SWA 1024 : global = 3:1**, RoPE θ=500k | Mellum2 arXiv:2605.31268 (3:1 SWA, θ=500k for code); Gemma 4 TR arXiv:2607.02770 (4:1–5:1, QKNorm); hybrid-attention survey arXiv:2606.15378, FlashMorph arXiv:2606.30562 for the linear-attention option |
| Optimizer | **Muon** 2e-3 (2D hidden matrices) + AdamW 3e-4 (embeddings/1D), **FP32 master weights**, warmup-hold-decay (5k / 40% hold / cosine → 3e-5) | `aletheia-lm/config.py`; arXiv:2505.02222; NVIDIA *SOAP, Muon, and Beyond* arXiv:2607.20548; `torch.optim.Muon` since PyTorch 2.9 |
| Token budget | knob, not a constant: **33 tok/param** reproduces the live run (8.19B, every token unique); **13,000 tok/param** is the LFM2.5 target and needs 3.2T tokens — i.e. a corpus ~190× the current 16.8B unique, which is why the data cell streams the Hub instead of only reading `local_repos` | LFM2.5-2.6B ≈ 34T tokens / 2.69B params ≈ 12.6–13k tok/param; repetition decay arXiv:2305.16264 (R*_D≈15) is what forbids reaching 13k by looping a 16.8B corpus |
| Data | Nemotron-CC-HQ + FineWeb-Edu + DCLM + code, **Aletheia repo upsampled**, chat anneal from SmolTalk2 | Nemotron-CC arXiv:2412.02595 (+5.6 MMLU over DCLM at 8B/1T); SmolLM3/SmolTalk2 recipe |
| Eval | lm-eval-harness v2 hard suite (MMLU-Pro, GPQA, BBH, MuSR, IFEval, MATH-L5) + tokenizer/throughput/precision-gap benchmarks + an Aletheia knowledge probe | EleutherAI lm-evaluation-harness |

**Hardware honesty (read this).** `check_nvfp4_support()` in Transformer Engine requires compute
capability ≥ 10.0. Datacenter Blackwell (B200/B300, sm_100) and consumer Blackwell (RTX 5070/5080/5090
= sm_120, RTX PRO 6000) both satisfy that inequality, but TE ships Linux wheels only — on Windows use
**WSL2**. The default config is `aletheia-lm`'s 280.8M model at seq 2048, which fits a 12 GB RTX 5070
comfortably; what a 5070 cannot do is 13k tok/param in human time (the MLX run needs 31 days for 33
tok/param). Every scale knob is a constant in one config cell. If NVFP4 is unavailable the notebook
degrades MXFP8 → FP8 → BF16 and tells you which path it took, so the same notebook is also a valid
CUDA BF16 port of the MLX trainer.
""")

# ---------------------------------------------------------------- 1. install
md(r"""
## 1. Environment

Latest releases as of 2026-08-08 (Linux / WSL2; CUDA 13.0 is the PyPI default, CUDA 13.2 —
`.../whl/nightly/cu132` — carries the widest Blackwell support; for source builds on RTX 50-series
include `12.0 12.1` in `TORCH_CUDA_ARCH_LIST`):

```bash
pip install --upgrade "torch>=2.13.0" --index-url https://download.pytorch.org/whl/cu130
pip install --no-build-isolation "transformer_engine[pytorch]>=2.17.1"
pip install "tokenizers>=0.23.1" "transformers>=5.14.1" "datasets>=5.0.1" "safetensors>=0.6" \
            "numpy>=2.1" "matplotlib>=3.10" "tqdm>=4.67" "zstandard>=0.24" ipywidgets
# optional
pip install flash-linear-attention lm-eval tiktoken sentencepiece
```

Older stacks also work: the notebook falls back from TE's `autocast` to the deprecated
`fp8_autocast`, and to a bundled Newton–Schulz Muon when `torch.optim.Muon` is absent (< 2.9).
""")

code(r"""
# ---- capability probe: decide the precision path before anything else -------
import importlib, math, os, platform, subprocess, sys

import torch

print(f"python      : {sys.version.split()[0]}  ({platform.system()})")
print(f"torch       : {torch.__version__}  cuda={torch.version.cuda}")

HAS_CUDA = torch.cuda.is_available()
CC = torch.cuda.get_device_capability() if HAS_CUDA else (0, 0)
GPU = torch.cuda.get_device_name() if HAS_CUDA else "cpu"
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_CUDA else 0.0
SM = CC[0] * 10 + CC[1]
print(f"device      : {GPU}  sm_{SM}  {VRAM_GB:.1f} GB")

TE = None
NVFP4_OK = MXFP8_OK = FP8_OK = False
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import (
        DelayedScaling, Float8CurrentScaling, MXFP8BlockScaling, NVFP4BlockScaling,
    )
    TE = te
    print(f"transformer_engine: {importlib.metadata.version('transformer_engine')}")
    # TE moved these helpers to transformer_engine.pytorch.quantization; fp8.py re-exports them.
    try:
        from transformer_engine.pytorch.quantization import (
            check_fp8_support, check_mxfp8_support, check_nvfp4_support,
        )
    except Exception:
        from transformer_engine.pytorch.fp8 import (
            check_fp8_support, check_mxfp8_support, check_nvfp4_support,
        )
    NVFP4_OK, nvfp4_why = check_nvfp4_support()
    MXFP8_OK, mxfp8_why = check_mxfp8_support()
    FP8_OK, fp8_why = check_fp8_support()
    print(f"NVFP4       : {NVFP4_OK}  {'' if NVFP4_OK else nvfp4_why}")
    print(f"MXFP8       : {MXFP8_OK}  {'' if MXFP8_OK else mxfp8_why}")
    print(f"FP8         : {FP8_OK}  {'' if FP8_OK else fp8_why}")
except Exception as exc:                                    # TE absent (e.g. native Windows)
    print(f"transformer_engine: unavailable -> {type(exc).__name__}: {exc}")
    print("            BF16 fallback will be used. Install TE under Linux/WSL2 for FP4.")

# NVFP4 needs cc>=10.0: sm_100 (B200/B300) and sm_120/121 (RTX 50, RTX PRO 6000) both qualify.
PRECISION_PATH = (
    "nvfp4" if NVFP4_OK else "mxfp8" if MXFP8_OK else "fp8" if FP8_OK else "bf16"
)
print(f"\n>> precision path: {PRECISION_PATH.upper()}")

HAS_FLA = importlib.util.find_spec("fla") is not None
HAS_LMEVAL = importlib.util.find_spec("lm_eval") is not None
print(f">> flash-linear-attention: {HAS_FLA} | lm-eval: {HAS_LMEVAL}")

if HAS_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
""")

# ---------------------------------------------------------------- 2. config
md(r"""
## 2. One config cell — `aletheia-lm`'s config, ported

Defaults are `aletheia-lm/src/aletheia_lm/config.py` verbatim, so a checkpoint from this notebook is
comparable with the live MLX run rather than being a different experiment. Only the precision block
and the FP32-master-weight repair are new.

`PRESET`: `parity` reproduces the MLX run's shape (280.8M, seq 2048, mb 2 × accum 4, 500k steps);
`laptop` shrinks it for a 12 GB smoke run; `overtrain` is the 13,000 tok/param target, which is a
*data* decision before it is a compute one — see the budget report the cell prints.
""")

code(r"""
from dataclasses import dataclass, field, asdict
from pathlib import Path

PRESET = "rtx5070"         # "rtx5070" | "workstation" | "cluster"

@dataclass
class Cfg:
    # ---- identity / paths
    name: str = "Aletheia-NVFP4"
    root: Path = Path("./aletheia_nvfp4")
    # Optional: a local aletheia-lm checkout, used ONLY as a tokenizer benchmark baseline.
    # Nothing is initialised from it -- that model is a code model, this one is a chat model.
    lm_repo: Path = Path("./aletheia-lm")

    # ---- tokenizer: trained here, from scratch (SuperBPE two-stage, byte-level)
    vocab_size: int = 32768          # 32k-class per TOKENIZER-V3's sweep + arXiv:2407.13623;
                                     # %128 for FP4 GEMM alignment, <2^16 so shards stay uint16
    superword_transition: float = 0.90   # stage 1 (pretokenised) share; stage 2 crosses whitespace
    tok_corpus_gb: float = 1.5       # fitting sample; 1.5 GB is what fits in RAM on this box
    superword_sample_mb: int = 512   # stage-2 fitting sample
    max_token_bytes: int = 48

    # ---- architecture: sized so pretraining actually completes on a 12 GB RTX 5070
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4              # GQA 4:1
    head_dim: int = 64               # d_model // n_heads
    d_ff: int = 2816                 # ~(8/3)*d_model, multiple of 256
    rms_eps: float = 1e-5
    tie_embeddings: bool = True
    qk_norm: bool = True             # RMSNorm on q/k; kills attention-logit blowup in low precision
    global_every: int = 4            # 3 sliding-window layers : 1 global attention layer
    window: int = 1024
    rope_local: float = 1e4          # Gemma 4 split bases: cheap local, long-range global
    rope_global: float = 1e6
    z_loss: float = 1e-4             # keeps the softmax denominator near 1
    # --- the August-2026 SOTA stack, each independently switchable ---------
    hybrid_linear: bool = True       # gated DeltaNet in the non-global slots when `fla` is present
    attn_sinks: bool = True          # learnable per-head sink logit (StreamingLLM / GPT-OSS)
    partial_rope: float = 0.25       # Gemma 4 pp-RoPE: rotate this fraction of dims on global layers
    intra_doc_mask: bool = True      # never attend across a document boundary inside a packed seq
    mtp_depth: int = 1               # multi-token prediction heads (DeepSeek-V3); 0 disables
    mtp_weight: float = 0.3
    moe: bool = False                # MoE FFN; memory-bound on 12 GB, on by default from
    moe_experts: int = 8             # `workstation` upward
    moe_top_k: int = 2
    moe_shared: int = 1              # always-on shared expert (DeepSeek-V3 / Qwen3)
    moe_bias_speed: float = 1e-3     # aux-loss-free load balancing via router bias nudging
    ema_decay: float = 0.0           # >0 keeps an EMA copy of the weights for the final checkpoint

    # ---- data: CHAT-first mixture (this is a conversational model, not a code model)
    seq_len: int = 2048
    micro_batch: int = 2
    grad_accum: int = 8
    aletheia_epochs: int = 24        # the OS repo is tiny: upsample it hard
    mix: dict = field(default_factory=lambda: {
        "nemotron_cc_hq": 0.34, "fineweb_edu": 0.20, "dclm": 0.08,
        "chat": 0.14, "code": 0.11, "math": 0.05, "aletheia": 0.08,
    })

    # ---- budget / schedule (warmup - hold - cosine decay)
    train_days: float = 14.0         # wall-clock budget; steps are derived from measured tok/s
    tokens_per_param: int = 13_000   # aspiration (LFM2.5 ratio). The cell reports what fits.
    total_steps: int = 0             # >0 pins the step count and ignores both knobs above
    warmup_frac: float = 0.01
    hold_frac: float = 0.40          # hold at peak after warmup, then cosine to lr_min
    lr_muon: float = 2.0e-3          # 2D hidden matrices
    lr_adamw: float = 3.0e-4         # embeddings / norms / 1D
    lr_min: float = 3.0e-5
    weight_decay: float = 0.1        # arXiv:2509.14786 suggests ~1.6 at this scale; unswept
    grad_clip: float = 1.0
    fp32_master: bool = True         # FP32 master weights, bf16/FP4 compute

    # ---- precision
    precision: str = "auto"          # auto | nvfp4 | mxfp8 | fp8 | bf16
    bf16_first_blocks: int = 2       # arXiv:2509.25149 keeps early+late blocks in BF16
    bf16_last_blocks: int = 4
    bf16_switch_frac: float = 0.82   # last 18% of tokens run in BF16
    nvfp4_rht: bool = True
    nvfp4_stochastic_rounding: bool = True
    nvfp4_2d_weights: bool = True
    fp4_attention: bool = False      # arXiv:2607.04422 full-stack FP4; off by default
    deterministic_hadamard: bool = False  # arXiv:2605.09825 remedy if Wgrad destabilises

    # ---- runtime
    compile: bool = True
    act_ckpt: bool = True
    save_every: int = 200
    eval_every: int = 200
    seed: int = 1337

cfg = Cfg()

if PRESET == "rtx5070":                 # 12 GB consumer Blackwell, the machine this was written on
    pass                                # defaults are already this shape
elif PRESET == "workstation":           # 48-96 GB (RTX PRO 6000 / H200)
    cfg.d_model, cfg.n_layers, cfg.head_dim = 1536, 28, 128
    cfg.n_heads, cfg.n_kv_heads, cfg.d_ff = 12, 4, 4096
    cfg.micro_batch, cfg.grad_accum = 8, 4
    cfg.moe, cfg.train_days = True, 30.0
elif PRESET == "cluster":               # multi-node B200
    cfg.d_model, cfg.n_layers, cfg.head_dim = 2048, 32, 128
    cfg.n_heads, cfg.n_kv_heads, cfg.d_ff = 16, 4, 5632
    cfg.seq_len, cfg.micro_batch, cfg.grad_accum = 4096, 16, 8
    cfg.moe, cfg.moe_experts, cfg.train_days = True, 32, 60.0

if cfg.precision == "auto":
    cfg.precision = PRECISION_PATH

D_FF = cfg.d_ff
for d in (cfg.d_model, D_FF, cfg.vocab_size, cfg.n_heads * cfg.head_dim,
          cfg.n_kv_heads * cfg.head_dim):
    assert d % 128 == 0, f"{d} must be a multiple of 128 for FP4/FP8 GEMM alignment"
assert cfg.n_heads % cfg.n_kv_heads == 0
assert cfg.vocab_size <= 65536, "uint16 shards require vocab <= 65536"

for sub in ("shards", "ckpt", "tokenizer", "corpus", "logs"):
    (cfg.root / sub).mkdir(parents=True, exist_ok=True)

torch.manual_seed(cfg.seed)

def param_estimate(c: Cfg, d_ff: int):
    emb = c.vocab_size * c.d_model
    attn = (c.d_model * c.n_heads * c.head_dim + 2 * c.d_model * c.n_kv_heads * c.head_dim
            + c.n_heads * c.head_dim * c.d_model)
    ffn = 3 * c.d_model * d_ff
    if c.moe:
        ffn = ffn * (c.moe_experts + c.moe_shared) + c.d_model * c.moe_experts
    per_layer = attn + ffn + 4 * c.d_model
    body = c.n_layers * per_layer + c.d_model
    body += c.mtp_depth * (attn + 3 * c.d_model * d_ff)          # MTP blocks
    active = c.n_layers * (attn + (3 * c.d_model * d_ff * (c.moe_top_k + c.moe_shared)
                                   if c.moe else 3 * c.d_model * d_ff))
    return emb + body + (0 if c.tie_embeddings else emb), body, active

TOTAL_PARAMS, NONEMB_PARAMS, ACTIVE_PARAMS = param_estimate(cfg, D_FF)
TOKENS_PER_STEP = cfg.seq_len * cfg.micro_batch * cfg.grad_accum

# Throughput prior, refined by the benchmark cell later. Blackwell BF16 tok/s per GFLOP-of-model,
# fitted to the published RTX 5090 / B200 TE numbers; only used to turn `train_days` into steps.
TPS_PRIOR = {"RTX 5070": 9_000, "RTX 5080": 16_000, "RTX 5090": 34_000,
             "RTX PRO 6000": 42_000, "B200": 190_000}
TPS_GUESS = next((v for k, v in TPS_PRIOR.items() if k.split()[-1] in GPU), 6_000)
TPS_GUESS = TPS_GUESS * (280e6 / max(ACTIVE_PARAMS, 1)) ** 0.9

BUDGET_ASPIRATION = cfg.tokens_per_param * NONEMB_PARAMS
BUDGET_WALLCLOCK = int(cfg.train_days * 86_400 * TPS_GUESS)
TOKEN_BUDGET = BUDGET_WALLCLOCK if not cfg.total_steps else cfg.total_steps * TOKENS_PER_STEP
TOTAL_STEPS = cfg.total_steps or max(1, TOKEN_BUDGET // TOKENS_PER_STEP)
WARMUP = max(1, int(TOTAL_STEPS * cfg.warmup_frac))

print(f"preset            : {PRESET}   precision: {cfg.precision}")
print(f"d_model/layers    : {cfg.d_model} x {cfg.n_layers}   d_ff={D_FF}"
      f"   heads {cfg.n_heads}/{cfg.n_kv_heads} x {cfg.head_dim}")
print(f"params            : {TOTAL_PARAMS/1e6:.1f}M total | {NONEMB_PARAMS/1e6:.1f}M non-embedding"
      f" | {ACTIVE_PARAMS/1e6:.1f}M active/token")
print(f"sota stack        : hybrid_linear={cfg.hybrid_linear} sinks={cfg.attn_sinks}"
      f" pp-rope={cfg.partial_rope} intra_doc={cfg.intra_doc_mask} mtp={cfg.mtp_depth}"
      f" moe={cfg.moe} ema={cfg.ema_decay or 'off'}")
print(f"tokens/step       : {TOKENS_PER_STEP:,}   steps: {TOTAL_STEPS:,}")
print(f"budget (wall)     : {TOKEN_BUDGET/1e9:.1f}B tokens in {cfg.train_days:.0f} days"
      f" at ~{TPS_GUESS:,.0f} tok/s  ->  {TOKEN_BUDGET/NONEMB_PARAMS:,.0f} tok/param")
print(f"BF16 switch at    : step {int(TOTAL_STEPS*cfg.bf16_switch_frac):,}")
print()
print(f"reality check     : {cfg.tokens_per_param:,} tok/param would need "
      f"{BUDGET_ASPIRATION/1e12:.2f}T tokens = {BUDGET_ASPIRATION/TPS_GUESS/86400/365:.1f} GPU-years"
      f" here.\n                    LFM2.5-2.6B spent ~34T tokens to reach that ratio. On this box "
      f"the honest\n                    target is the wall-clock budget above; raise `train_days`, "
      f"or move the same\n                    notebook to a `cluster` preset to close the gap.")
""")

# ---------------------------------------------------------------- 3. tokenizer
md(r"""
## 3. Tokenizer — a superword tokenizer for prose + Rust + Aletheia jargon

The old `aletheiacode64k` was a SentencePiece **unigram** model trained on code only: byte-fallback,
`character_coverage=1.0`, 64000 pieces. For a *conversational* model that is two problems — no
natural-language statistics in the vocabulary, and no whitespace-crossing pieces.

**SOTA choice: SuperBPE** (arXiv:2503.13423). Standard BPE forbids merges across the pretokeniser's
whitespace boundaries, so a token can never be `def main(`, `unsafe fn `, `it is`. SuperBPE trains in
two stages with a *transition point* `t`:

1. **stage 1 (`t` merges)** — ordinary byte-level BPE *with* whitespace pretokenisation: learns
   subwords.
2. **stage 2 (`V − t` merges)** — pretokenisation switched **off**: learns *superwords* that span
   spaces. Inference also runs with pretokenisation off, so the whole merge list applies.

At 8.1B params this gave +4.0 points average over 30 tasks (+8.2 MMLU) and −27% FLOPs per input byte
at the *same* vocabulary size. BoundlessBPE (arXiv:2504.00178) reaches the same conclusion from the
diminishing-returns-of-large-vocab angle.

Stage 2 is not exposed by `tokenizers`, so we learn it directly: encode the corpus with the stage-1
model, count adjacent pairs over the *ID stream* (crossing spaces is now allowed), greedily merge the
top pair, repeat. That is exactly SuperBPE's algorithm; the fast-trainer paper (arXiv:2604.05192)
shows the same greedy single-pass structure is what makes it tractable. The result is written back as
extra `merges` on a plain HF `BPE` model with a `ByteLevel`-only pretokeniser, so it loads anywhere
`transformers` runs, and byte-level coverage means **zero UNK** on arbitrary bytes.

Vocabulary is **65536**: divisible by 128 (FP4/FP8 GEMM alignment) and every id fits `uint16`, which
halves shard size versus the old `uint32` layout.
""")

code(r"""
# ---- 3a. corpus: Aletheia OS repo + prose + code, capped at cfg.tok_corpus_gb -----
import io, json, random, re, shutil, subprocess, time
from pathlib import Path

CORPUS = cfg.root / "corpus" / "tokenizer_corpus.txt"
ALETHEIA_DIR = cfg.root / "corpus" / "aletheia"

ALETHEIA_URL = "https://github.com/hotocoo/aletheia"
CODE_EXT = {
    ".rs", ".toml", ".c", ".h", ".cpp", ".hpp", ".cc", ".hh", ".s", ".S", ".asm",
    ".ld", ".lds", ".sh", ".py", ".wat", ".wit", ".json", ".yaml", ".yml", ".mk",
    ".cmake", ".dts", ".dtsi", ".proto", ".sql", ".ts", ".js",
}
DOC_EXT = {".md", ".rst", ".txt", ".adoc"}
SPECIAL_NAMES = {"Makefile", "Kconfig", "CMakeLists.txt", "Cargo.toml", "Cargo.lock", "justfile"}
SKIP_DIRS = {".git", "target", "build", "dist", "out", "node_modules", ".cache", ".venv"}

def clone_aletheia() -> Path:
    if ALETHEIA_DIR.exists():
        return ALETHEIA_DIR
    ALETHEIA_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", ALETHEIA_URL, str(ALETHEIA_DIR)], check=True)
    return ALETHEIA_DIR

def iter_repo_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.name in SPECIAL_NAMES or p.suffix in CODE_EXT or p.suffix in DOC_EXT:
            if p.stat().st_size < 4 << 20:
                yield p

def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    return "\n".join(l for l in text.splitlines() if len(l) < 20_000)

def build_tokenizer_corpus(target_gb: float) -> Path:
    if CORPUS.exists() and CORPUS.stat().st_size > 0.9 * target_gb * 1e9:
        print(f"[=] reuse {CORPUS} ({CORPUS.stat().st_size/1e9:.2f} GB)")
        return CORPUS
    budget = int(target_gb * 1e9)
    # Aletheia OS gets an outsized share of the tokenizer corpus on purpose: its identifiers
    # (`CapabilityRef`, `ContextFabric`, `IntentEnvelope`) must be single tokens.
    quota = {"aletheia": 0.12, "prose": 0.58, "code": 0.22, "math": 0.08}
    written = {k: 0 for k in quota}
    t0 = time.time()
    with open(CORPUS, "w", encoding="utf-8") as out:
        repo = clone_aletheia()
        for p in iter_repo_files(repo):
            if written["aletheia"] > quota["aletheia"] * budget:
                break
            try:
                txt = clean(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            rec = f"<|file|>{p.relative_to(repo)}\n{txt}\n\n"
            out.write(rec); written["aletheia"] += len(rec.encode())
        print(f"[+] aletheia: {written['aletheia']/1e6:.1f} MB")

        try:
            from datasets import load_dataset
            streams = {
                "prose": [
                    ("nvidia/Nemotron-CC-v2", None, "text"),
                    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
                ],
                "code": [("bigcode/the-stack-v2-dedup", None, "content")],
                "math": [("open-web-math/open-web-math", None, "text")],
            }
            for group, specs in streams.items():
                cap = quota[group] * budget
                for repo_id, sub, field in specs:
                    if written[group] >= cap:
                        break
                    try:
                        ds = load_dataset(repo_id, sub, split="train", streaming=True)
                    except Exception as exc:
                        print(f"[!] {repo_id}: {type(exc).__name__} -> skipped")
                        continue
                    for row in ds:
                        txt = row.get(field) or ""
                        if not txt:
                            continue
                        out.write(txt + "\n\n"); written[group] += len(txt) + 2
                        if written[group] >= cap:
                            break
                print(f"[+] {group}: {written[group]/1e6:.1f} MB")
        except ImportError:
            print("[!] `datasets` missing - tokenizer corpus is Aletheia-only (fine for a smoke run)")
    print(f"[=] corpus {CORPUS.stat().st_size/1e9:.2f} GB in {time.time()-t0:.0f}s")
    return CORPUS

CORPUS = build_tokenizer_corpus(cfg.tok_corpus_gb)
""")

code(r"""
# ---- 3b. special tokens: chat, tool use, and Aletheia OS structure ----------
SPECIALS = [
    "<|pad|>", "<|bos|>", "<|eos|>",
    # chat surface
    "<|system|>", "<|user|>", "<|assistant|>", "<|think|>", "<|/think|>",
    "<|tool_call|>", "<|tool_result|>", "<|eot|>",
    # Aletheia OS domain structure (mirrors intent -> plan -> capability -> policy -> execute)
    "<|intent|>", "<|context|>", "<|plan|>", "<|capability|>", "<|policy|>",
    "<|action|>", "<|provenance|>", "<|memory|>", "<|entity|>", "<|component|>",
    # source structure
    "<|file|>", "<|code|>", "<|diff|>", "<|adr|>", "<|doc|>",
]
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
assert len(SPECIALS) < 64
print(f"{len(SPECIALS)} special tokens; reserving 64 slots at the bottom of the vocab")
""")

code(r"""
# ---- 3c. SuperBPE trainer: stage 1 (pretokenised) + stage 2 (whitespace-crossing) ----
from collections import Counter

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

# Stage-1 split regex. This is the `xw` variant measured in aletheia-lm/docs/TOKENIZER-V3.md:
# a leading whitespace *run* attaches to the token that follows, so `\n    return` and `);\n` can
# become single pieces. That change alone was worth +8.6% bytes/token at fixed vocabulary there --
# the largest single tokenizer effect in that sweep. Numeric literals are also kept whole.
GPT4_SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|(?i:0x[0-9a-f][0-9a-f_]*|0b[01][01_]*|0o[0-7][0-7_]*)"
    r"|\s*[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\s*\p{N}{1,3}"
    r"|\s*[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s+"
)

STAGE1_PATH = cfg.root / "tokenizer" / "stage1.json"
FINAL_PATH = cfg.root / "tokenizer" / "tokenizer.json"

RESERVED = 64
T_MERGES = int(cfg.vocab_size * cfg.superword_transition)   # transition point t


def train_stage1(corpus: Path, vocab_t: int) -> Tokenizer:
    if STAGE1_PATH.exists():
        print(f"[=] reuse {STAGE1_PATH}")
        return Tokenizer.from_file(str(STAGE1_PATH))
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    # ByteLevel gives complete byte coverage (no UNK ever); the Split keeps stage 1 inside words.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=Regex(GPT4_SPLIT_PATTERN),
            behavior="isolated",
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_t,
        min_frequency=2,
        special_tokens=SPECIALS + [f"<|reserved_{i}|>" for i in range(RESERVED - len(SPECIALS))],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=cfg.max_token_bytes,
        show_progress=True,
    )
    t0 = time.time()
    tok.train([str(corpus)], trainer)
    print(f"[+] stage 1: {tok.get_vocab_size()} pieces in {time.time()-t0:.0f}s")
    tok.save(str(STAGE1_PATH))
    return tok


def learn_superword_merges(stage1: Tokenizer, corpus: Path, n_merges: int,
                           sample_bytes: int = 400_000_000, min_freq: int = 4):
    # Stage 2: pretokenisation OFF. Encode with stage 1, then greedily merge the most frequent
    # adjacent pair over the *id stream*, so merges may cross whitespace -> superwords.
    #
    # Naive re-counting is O(n_merges x corpus) and dies on real corpora (this is the 4.7-CPU-day
    # problem in arXiv:2504.00178). Instead: a doubly linked list over the id stream + per-pair
    # position sets + a lazy max-heap, so each merge costs O(occurrences of that pair). This is the
    # incremental structure of arXiv:2604.05192.
    import heapq
    from array import array
    from collections import defaultdict

    id2tok = {i: t for t, i in stage1.get_vocab().items()}
    vocab_next = stage1.get_vocab_size()

    toks, read = array("i"), 0
    with open(corpus, encoding="utf-8") as f:
        buf, buflen = [], 0
        for line in f:
            buf.append(line); buflen += len(line); read += len(line)
            if buflen > 8 << 20:
                toks.extend(stage1.encode("".join(buf), add_special_tokens=False).ids)
                buf, buflen = [], 0
            if read > sample_bytes:
                break
        if buf:
            toks.extend(stage1.encode("".join(buf), add_special_tokens=False).ids)
    n = len(toks)
    print(f"[*] stage 2 corpus: {read/1e6:.0f} MB -> {n/1e6:.2f}M ids")

    prv = array("l", range(-1, n - 1))
    nxt = array("l", range(1, n + 1)); nxt[n - 1] = -1
    alive = bytearray(b"\x01") * n

    special = set(range(RESERVED))
    counts = Counter()
    where = defaultdict(set)

    def ok(a, b):
        return a not in special and b not in special

    for i in range(n - 1):
        a, b = toks[i], toks[i + 1]
        if ok(a, b):
            counts[(a, b)] += 1
            where[(a, b)].add(i)

    heap = [(-c, p) for p, c in counts.items()]
    heapq.heapify(heap)

    def bump(pair, i, delta):
        if not ok(*pair):
            return
        counts[pair] += delta
        if delta > 0:
            where[pair].add(i)
            heapq.heappush(heap, (-counts[pair], pair))
        else:
            where[pair].discard(i)

    merges, t0, done = [], time.time(), 0
    while done < n_merges and heap:
        negc, pair = heapq.heappop(heap)
        if counts.get(pair, 0) != -negc:
            continue                                   # stale heap entry
        if -negc < min_freq:
            print(f"[!] stopping at {done} superword merges (best pair freq {-negc})")
            break
        a, b = pair
        piece = id2tok[a] + id2tok[b]
        if len(piece.encode()) > cfg.max_token_bytes:
            counts[pair] = 0; where.pop(pair, None)    # too long: retire this pair
            continue
        new_id = vocab_next; vocab_next += 1
        id2tok[new_id] = piece
        merges.append((id2tok[a], id2tok[b]))

        for i in sorted(where.pop(pair, ())):
            j = nxt[i]
            if not alive[i] or j == -1 or not alive[j] or toks[i] != a or toks[j] != b:
                continue
            p, q = prv[i], nxt[j]
            if p != -1:
                bump((toks[p], a), p, -1)
            if q != -1:
                bump((b, toks[q]), j, -1)
            toks[i] = new_id
            alive[j] = 0
            nxt[i] = q
            if q != -1:
                prv[q] = i
            if p != -1:
                bump((toks[p], new_id), p, +1)
            if q != -1:
                bump((new_id, toks[q]), i, +1)
        counts[pair] = 0
        done += 1
        if done % 1024 == 0:
            print(f"    {done}/{n_merges} superwords  last={piece!r}  {time.time()-t0:.0f}s")
    print(f"[+] {len(merges)} superword merges in {time.time()-t0:.0f}s")
    return merges, id2tok


def assemble_final(stage1: Tokenizer, merges, id2tok) -> Tokenizer:
    state = json.loads(stage1.to_str())
    vocab = dict(state["model"]["vocab"])
    for i in sorted(id2tok):
        vocab.setdefault(id2tok[i], i)
    state["model"]["vocab"] = vocab
    state["model"]["merges"] = list(state["model"]["merges"]) + [list(m) for m in merges]
    # inference with pretokenisation OFF -> the superword merges can fire
    state["pre_tokenizer"] = {
        "type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False,
    }
    tok = Tokenizer.from_str(json.dumps(state))
    tok.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A", pair="<|bos|> $A <|bos|> $B:1",
        special_tokens=[("<|bos|>", BOS_ID)],
    )
    tok.decoder = decoders.ByteLevel()
    tok.save(str(FINAL_PATH))
    return tok


stage1 = train_stage1(CORPUS, T_MERGES)
sw_merges, id2tok = learn_superword_merges(stage1, CORPUS, cfg.vocab_size - T_MERGES)
tokenizer = assemble_final(stage1, sw_merges, id2tok)
VOCAB_SIZE_ACTUAL = tokenizer.get_vocab_size()
print(f"[=] final vocab {VOCAB_SIZE_ACTUAL} (target {cfg.vocab_size})")
""")

code(r"""
# ---- 3d. wrap as a HF tokenizer with an Aletheia chat template --------------
from transformers import PreTrainedTokenizerFast

CHAT_TEMPLATE = (
    "{{ '<|bos|>' }}"
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}{{ '<|system|>' + m['content'] + '<|eot|>' }}"
    "{% elif m['role'] == 'user' %}{{ '<|user|>' + m['content'] + '<|eot|>' }}"
    "{% elif m['role'] == 'tool' %}{{ '<|tool_result|>' + m['content'] + '<|eot|>' }}"
    "{% else %}{{ '<|assistant|>' + m['content'] + '<|eot|>' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)

hf_tok = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|bos|>", eos_token="<|eos|>", pad_token="<|pad|>",
    additional_special_tokens=[s for s in SPECIALS if s not in ("<|bos|>", "<|eos|>", "<|pad|>")],
    model_max_length=cfg.seq_len,
    chat_template=CHAT_TEMPLATE,
)
hf_tok.save_pretrained(cfg.root / "tokenizer")
print(hf_tok.apply_chat_template(
    [{"role": "system", "content": "You are Aletheia."},
     {"role": "user", "content": "Explain the capability engine."}],
    tokenize=False, add_generation_prompt=True))
""")

md(r"""
### 3e. Tokenizer benchmark

Fewer tokens per byte is a *direct* multiplier on training throughput and on effective context: the
SuperBPE paper's −27% FLOPs/byte came entirely from this number. We measure bytes/token on prose,
Rust, C, assembly and Aletheia docs, against the stage-1 (plain BPE) baseline, the old
`aletheiacode64k` unigram model if present, and `cl100k_base`/`o200k_base` if `tiktoken` is
installed. Also asserted: **byte-exact round-trip** and **zero UNK**.
""")

code(r"""
SAMPLES = {
    "prose": (
        "The capability engine issues unforgeable, attenuable, revocable authority. A component "
        "that holds a capability may narrow it before delegation, but never widen it."
    ),
    "chat": "<|user|>How does Aletheia validate a model proposal before execution?<|eot|>",
    "rust": (
        "pub fn evaluate(&self, intent: &Intent, ctx: &Context) -> Result<Decision, PolicyError> {\n"
        "    let caps = self.capabilities.resolve(intent.subject())?;\n"
        "    if !caps.permits(intent.action()) { return Err(PolicyError::Denied); }\n"
        "    Ok(Decision::Approve)\n}\n"
    ),
    "c": "#include <stdint.h>\nstatic inline void wrmsr(uint32_t msr, uint64_t v) {\n"
         "    asm volatile(\"wrmsr\" :: \"c\"(msr), \"a\"((uint32_t)v), \"d\"((uint32_t)(v >> 32)));\n}\n",
    "asm": "section .text\nglobal _start\n_start:\n    mov rax, 60\n    xor rdi, rdi\n    syscall\n",
    "adr": "# ADR-0007: Policy approval is independent of capability authority\n\nStatus: Accepted\n",
}

def bench(name, encode):
    rows = []
    for k, text in SAMPLES.items():
        ids = encode(text)
        rows.append(len(text.encode()) / max(1, len(ids)))
    return name, rows

encoders = [
    ("SuperBPE 65k (ours)", lambda t: tokenizer.encode(t, add_special_tokens=False).ids),
    ("stage-1 BPE (no superwords)", lambda t: stage1.encode(t, add_special_tokens=False).ids),
]
v3_json = cfg.lm_repo / "test" / "aletheia_tok_v3" / "tokenizer.json"
if v3_json.exists():                       # baseline only -- nothing is initialised from it
    _v3 = Tokenizer.from_file(str(v3_json))
    encoders.append(("aletheia_tok_v3 32k xw", lambda t: _v3.encode(t, add_special_tokens=False).ids))
old_sp = Path("aletheiacode64k.model")
if old_sp.exists():
    import sentencepiece as spm
    _sp = spm.SentencePieceProcessor(); _sp.load(str(old_sp))
    encoders.append(("aletheiacode64k unigram", lambda t: _sp.encode(t)))
try:
    import tiktoken
    for enc_name in ("cl100k_base", "o200k_base"):
        _e = tiktoken.get_encoding(enc_name)
        encoders.append((enc_name, lambda t, _e=_e: _e.encode(t)))
except Exception:
    pass

hdr = f"{'tokenizer':30s}" + "".join(f"{k:>10s}" for k in SAMPLES) + f"{'mean':>10s}"
print(hdr); print("-" * len(hdr))
for name, enc in encoders:
    _, rows = bench(name, enc)
    print(f"{name:30s}" + "".join(f"{v:10.3f}" for v in rows) + f"{sum(rows)/len(rows):10.3f}")
print("\n(bytes per token - higher is better)")

for k, text in SAMPLES.items():
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    back = tokenizer.decode(ids, skip_special_tokens=False)   # chat markers must survive
    assert back == text, f"round-trip FAIL on {k}:\n  in : {text!r}\n  out: {back!r}"
print("round-trip: PASS on all samples (byte-level -> no UNK possible)")

demo = tokenizer.encode("    let caps = self.capabilities.resolve(", add_special_tokens=False)
print("\nsuperword example:", demo.tokens)
""")

# ---------------------------------------------------------------- 4. data
md(r"""
## 4. Data pipeline — mixture, upsampling, `uint16` shards

Mixture rationale (Aug 2026):

* **Nemotron-CC-HQ** (arXiv:2412.02595) — classifier-ensemble + synthetic rephrasing; +5.6 MMLU over
  DCLM at 8B/1T, and 4× more unique real tokens than DCLM at full size. Largest single share.
* **FineWeb-Edu** + **DCLM** — different filters, so their union de-correlates filter bias
  (the Zyda-2 argument).
* **code + math** — the target model reads and writes Rust/C/asm about a kernel.
* **Aletheia OS repo** — a few MB, upsampled `aletheia_epochs` times. This is the knowledge the model
  must actually hold, and repetition is precisely what the 13k tok/param regime licenses.
* **chat** — a small share during pretraining so the chat special tokens are never cold at anneal.

Shards are flat `uint16` (vocab ≤ 65536), documents separated by `<|eos|>`, plus a `.idx` of document
offsets so the loader can respect document boundaries. This is the same memmap-shard pattern as
`aletheiaMLX.ipynb`, halved in size and with an index.
""")

code(r"""
import numpy as np

SHARDS = cfg.root / "shards"
SHARD_TOKENS = 200_000_000          # 400 MB per shard at uint16
DTYPE = np.uint16
assert VOCAB_SIZE_ACTUAL <= 65536

class ShardWriter:
    def __init__(self, out: Path, split: str):
        self.out, self.split = out, split
        self.buf = np.empty(SHARD_TOKENS, dtype=DTYPE)
        self.pos, self.sid, self.total = 0, 0, 0
        self.doc_offsets = []

    def add(self, ids):
        ids = np.asarray(ids, dtype=DTYPE)
        self.doc_offsets.append(self.sid * SHARD_TOKENS + self.pos)
        off, rem = 0, len(ids)
        while rem:
            take = min(SHARD_TOKENS - self.pos, rem)
            self.buf[self.pos:self.pos + take] = ids[off:off + take]
            self.pos += take; off += take; rem -= take; self.total += take
            if self.pos == SHARD_TOKENS:
                self._flush()

    def _flush(self):
        p = self.out / f"{self.split}_{self.sid:05d}.bin"
        self.buf[:self.pos].tofile(p)
        self.sid += 1; self.pos = 0

    def close(self):
        if self.pos:
            self._flush()
        np.asarray(self.doc_offsets, dtype=np.int64).tofile(self.out / f"{self.split}.idx")
        meta = {"tokens": self.total, "shards": self.sid, "dtype": "uint16",
                "docs": len(self.doc_offsets), "vocab": VOCAB_SIZE_ACTUAL}
        (self.out / f"{self.split}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[=] {self.split}: {self.total:,} tokens / {self.sid} shards / {len(self.doc_offsets):,} docs")
        return meta


def encode_doc(text: str, prefix: str = ""):
    ids = tokenizer.encode(prefix + text, add_special_tokens=False).ids
    return ids + [EOS_ID]


def aletheia_documents():
    repo = clone_aletheia()
    docs = []
    for p in iter_repo_files(repo):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not txt.strip():
            continue
        tag = "<|adr|>" if "adr" in str(p).lower() else (
            "<|doc|>" if p.suffix in DOC_EXT else "<|code|>")
        docs.append(f"{tag}<|file|>{p.relative_to(repo)}\n{txt}")
    print(f"[+] aletheia docs: {len(docs)}")
    return docs


STREAM_SPECS = {
    "nemotron_cc_hq": ("nvidia/Nemotron-CC-v2", None, "text"),
    "fineweb_edu":    ("HuggingFaceFW/fineweb-edu", "sample-100BT", "text"),
    "dclm":           ("mlfoundations/dclm-baseline-1.0-parquet", None, "text"),
    "code":           ("bigcode/the-stack-v2-dedup", None, "content"),
    "math":           ("open-web-math/open-web-math", None, "text"),
    "chat":           ("HuggingFaceTB/smoltalk2", "SFT", None),
}

def chat_to_text(row):
    msgs = row.get("messages") or row.get("conversations")
    if not msgs:
        return None
    try:
        return hf_tok.apply_chat_template(msgs, tokenize=False)
    except Exception:
        return None


def build_shards(target_tokens: int, val_tokens: int = 4_000_000, force: bool = False):
    meta_path = SHARDS / "train_meta.json"
    if meta_path.exists() and not force:
        print(f"[=] reuse shards: {json.loads(meta_path.read_text())}")
        return
    for f in SHARDS.glob("*"):
        f.unlink()
    train, val = ShardWriter(SHARDS, "train"), ShardWriter(SHARDS, "val")
    quotas = {k: int(v * target_tokens) for k, v in cfg.mix.items()}
    print("token quotas:", {k: f"{v/1e6:.0f}M" for k, v in quotas.items()})

    # --- Aletheia: the whole repo, repeated
    ale_docs = aletheia_documents()
    if ale_docs:
        written = 0
        for epoch in range(cfg.aletheia_epochs):
            random.Random(cfg.seed + epoch).shuffle(ale_docs)
            for d in ale_docs:
                ids = encode_doc(d)
                (val if written == 0 and epoch == 0 and random.random() < 0.02 else train).add(ids)
                written += len(ids)
                if written >= quotas.get("aletheia", 0):
                    break
            if written >= quotas.get("aletheia", 0):
                break
        print(f"[+] aletheia: {written/1e6:.1f}M tokens ({cfg.aletheia_epochs} epochs max)")

    # --- streamed corpora
    try:
        from datasets import load_dataset
    except ImportError:
        print("[!] `datasets` missing - training on Aletheia only")
        train.close(); val.close(); return

    from tqdm.auto import tqdm
    for key, quota in quotas.items():
        if key == "aletheia" or quota <= 0:
            continue
        repo_id, sub, field = STREAM_SPECS[key]
        try:
            ds = load_dataset(repo_id, sub, split="train", streaming=True)
        except Exception as exc:
            print(f"[!] {repo_id}: {type(exc).__name__} -> skipped"); continue
        got, val_got = 0, 0
        bar = tqdm(total=quota, unit="tok", desc=key, dynamic_ncols=True)
        for row in ds:
            text = chat_to_text(row) if key == "chat" else row.get(field)
            if not text:
                continue
            ids = encode_doc(text)
            if val_got < val_tokens * cfg.mix[key]:
                val.add(ids); val_got += len(ids)
            else:
                train.add(ids); got += len(ids); bar.update(len(ids))
            if got >= quota:
                break
        bar.close()
    train.close(); val.close()

build_shards(TOKEN_BUDGET)
""")

code(r"""
# ---- 4b. loader: packed sequences, document-boundary aware, infinite ---------
class PackedShards:
    def __init__(self, split: str, seq_len: int, batch: int, seed: int = 0):
        meta = json.loads((SHARDS / f"{split}_meta.json").read_text())
        self.arrs = [np.memmap(SHARDS / f"{split}_{i:05d}.bin", dtype=DTYPE, mode="r")
                     for i in range(meta["shards"])]
        self.sizes = [len(a) for a in self.arrs]
        self.seq_len, self.batch = seq_len, batch
        self.blocks = [(s, i) for s, n in enumerate(self.sizes)
                       for i in range((n - 1) // seq_len)]
        self.rng = np.random.default_rng(seed)
        self.tokens = meta["tokens"]
        print(f"[=] {split}: {self.tokens:,} tokens, {len(self.blocks):,} blocks of {seq_len}")

    def __iter__(self):
        order = self.rng.permutation(len(self.blocks))
        xb, yb = [], []
        for j in order:
            s, i = self.blocks[j]
            chunk = np.asarray(self.arrs[s][i * self.seq_len: i * self.seq_len + self.seq_len + 1],
                               dtype=np.int64)
            xb.append(chunk[:-1]); yb.append(chunk[1:])
            if len(xb) == self.batch:
                yield (torch.from_numpy(np.stack(xb)), torch.from_numpy(np.stack(yb)))
                xb, yb = [], []

def infinite(loader):
    while True:
        yield from loader

train_loader = PackedShards("train", cfg.seq_len, cfg.micro_batch, seed=cfg.seed)
val_loader = PackedShards("val", cfg.seq_len, cfg.micro_batch, seed=cfg.seed + 1)
train_iter = infinite(train_loader)
""")

# ---------------------------------------------------------------- 5. model
md(r"""
## 5. Architecture

Every architectural component that was state of the art in August 2026 and is compatible with 4-bit
training is here and **on by default**, each behind its own flag so any one of them can be ablated:

| component | what it does | source |
|---|---|---|
| **Hybrid attention 3:1** | 3 efficient layers per full-attention layer. Efficient = **gated DeltaNet** (linear attention) when `flash-linear-attention` is installed, otherwise **sliding-window attention** (1024) | Qwen3-Next / Nemotron-Flash arXiv:2511.18890 / MiniCPM-SALA; ratio from Gemma 4 arXiv:2607.02770 (4:1–5:1) and Mellum2 arXiv:2605.31268 (3:1). Layer *placement* matters more than the ratio — FlashMorph arXiv:2606.30562 — so the full-attention layers are evenly spaced, the one placement that survives every study |
| **GQA 4:1** | 16 query heads, 4 KV heads | KV cache is the binding constraint for an OS-resident chat model |
| **QK-norm** | RMSNorm on q and k | Gemma 3/4. Replaces logit soft-capping, which costs ~15% throughput in the TE nanochat measurements, and bounds attention logits — worth more in 4-bit than in BF16 |
| **Attention sinks** | learnable per-head sink key with a zero value: it takes softmax mass without contributing to the output | StreamingLLM / GPT-OSS. This is what makes windowed attention stable when the window slides past the prompt |
| **pp-RoPE** | rotate only the first 25% of head dims on **global** layers, full RoPE at θ=10k on local layers, θ=1M on global | Gemma 4's `p=0.25` split-base scheme. Partial rotation leaves clean unrotated channels for long-range content |
| **Pre + post norm** | RMSNorm on both sides of every sublayer | Gemma 4. The residual-stream stabiliser that keeps 4-bit forward passes from drifting |
| **SwiGLU / no biases / tied embeddings** | — | universal in 2026 dense stacks |
| **MoE FFN** (`cfg.moe`) | top-2 of 8 routed experts + 1 always-on shared expert, **aux-loss-free** load balancing by nudging per-expert router bias | DeepSeek-V3 bias-balancing, Qwen3/Gemma 4 26B-A4B shared-expert shape. Off on 12 GB — MoE trades memory for FLOPs and memory is what a 5070 lacks — on from the `workstation` preset up |
| **Multi-token prediction** | one extra block predicting `t+2`, loss weight 0.3; also gives free speculative decoding at inference | DeepSeek-V3 MTP. A data-efficiency win, which is exactly what an overtrained run wants |
| **Intra-document masking** | tokens never attend across an `<\|eos\|>` boundary inside a packed sequence | standard 2025-26 pretraining hygiene; packing without it teaches spurious cross-document dependence |
| **z-loss 1e-4 + FP32 logits** | keeps `logsumexp` bounded | logits are never FP4 |
| **Depth-scaled init** | residual projections initialised at `d_model^-0.5 / sqrt(2·n_layers)` | GPT-NeoX / OLMo |
| **Precision policy** | first 2 + last 4 blocks BF16; embeddings, LM head, norms, attention, optimizer state never FP4 | arXiv:2509.25149's 16%-BF16 rule |

**Deliberately declined**, with reasons, because "everything SOTA" also means not stacking things
that fight each other: full sparse attention (InfLLM-v2) — its win is at ≥128k context, which this
model does not train at; Mamba-2/SSM blocks — gated DeltaNet already occupies that slot and TE's FP4
path covers `Linear`, not scan kernels; hyper-connections and value-residual variants — promising but
each one invalidates the depth-scaled init that the 4-bit run depends on; MLA (DeepSeek-style latent
KV) — a real KV-cache win, but it interacts with pp-RoPE in ways nobody has published at this scale.
""")

code(r"""
import torch.nn as nn
import torch.nn.functional as F

BF16 = torch.bfloat16

def linear_factory(kind: str, layer_idx: int):
    # kind: 'attn' | 'ffn'; returns a constructor for a bias-free Linear in the right precision.
    quantizable = (
        cfg.precision in ("nvfp4", "mxfp8", "fp8")
        and TE is not None
        and cfg.bf16_first_blocks <= layer_idx < cfg.n_layers - cfg.bf16_last_blocks
    )
    if quantizable:
        return lambda i, o: TE.Linear(i, o, bias=False, params_dtype=BF16)
    return lambda i, o: nn.Linear(i, o, bias=False, dtype=BF16)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d, dtype=BF16)); self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt) * self.w)


def build_masks(idx, device):
    # One mask set per forward, shared by every layer. Combines: causal, sliding window,
    # intra-document (never attend across <|eos|>), and a leading always-visible sink column.
    B, L = idx.shape
    i = torch.arange(L, device=device)
    causal = i[:, None] >= i[None, :]
    local = causal & ((i[:, None] - i[None, :]) < cfg.window)
    if cfg.intra_doc_mask:
        doc = (idx == EOS_ID).cumsum(-1)                       # document id per position
        doc = doc - (idx == EOS_ID).long()                     # the EOS itself ends its own doc
        same = doc[:, :, None] == doc[:, None, :]              # (B, L, L)
        causal = causal[None] & same
        local = local[None] & same
    else:
        causal, local = causal[None].expand(B, L, L), local[None].expand(B, L, L)
    if cfg.attn_sinks:                                          # sink column is always visible
        ones = torch.ones(B, L, 1, dtype=torch.bool, device=device)
        causal = torch.cat([ones, causal], -1)
        local = torch.cat([ones, local], -1)
    return causal[:, None], local[:, None]                      # (B, 1, L, L(+1))


def rope_cache(seq: int, dim: int, base: float, device, dtype=torch.float32):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(seq, device=device, dtype=dtype)
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin, frac: float = 1.0):
    # x: (B, H, L, D). Rotation runs in fp32 (cos/sin are fp32) and comes back in x's dtype so
    # q/k/v stay type-consistent for SDPA. `frac` < 1 is Gemma 4's pp-RoPE: only the first
    # `frac` of the head dimension is rotated, the rest passes through unrotated.
    dt = x.dtype
    d = x.shape[-1]
    r = d if frac >= 1.0 else max(2, int(d * frac) // 2 * 2)
    xr, xp = x[..., :r].float(), x[..., r:]
    x1, x2 = xr[..., ::2], xr[..., 1::2]
    c, s = cos[None, None, :, : r // 2], sin[None, None, :, : r // 2]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.stack((o1, o2), dim=-1).flatten(-2).to(dt)
    return out if r == d else torch.cat([out, xp], -1)


class Attention(nn.Module):
    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx
        self.is_global = (idx + 1) % cfg.global_every == 0
        self.window = None if self.is_global else cfg.window
        self.nh, self.nkv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.rope_frac = cfg.partial_rope if self.is_global else 1.0   # Gemma 4 pp-RoPE
        L = linear_factory("attn", idx)
        self.q_proj = L(cfg.d_model, self.nh * self.hd)
        self.k_proj = L(cfg.d_model, self.nkv * self.hd)
        self.v_proj = L(cfg.d_model, self.nkv * self.hd)
        self.o_proj = L(self.nh * self.hd, cfg.d_model)
        self.q_norm = RMSNorm(self.hd, cfg.rms_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.hd, cfg.rms_eps) if cfg.qk_norm else nn.Identity()
        # Attention sink: a learnable key with a hard-zero value. It absorbs softmax mass that
        # would otherwise be forced onto real tokens when the window slides past the prompt.
        self.sink_k = nn.Parameter(torch.zeros(1, self.nkv, 1, self.hd, dtype=BF16)) \
            if cfg.attn_sinks else None

    def forward(self, x, rope, masks, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nkv, self.hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope[1 if self.is_global else 0]
        off = 0 if cache is None or cache[0] is None else cache[0].shape[2]
        q = apply_rope(q, cos[off:off + L], sin[off:off + L], self.rope_frac)
        k = apply_rope(k, cos[off:off + L], sin[off:off + L], self.rope_frac)
        new_cache = None
        if cache is not None:
            if cache[0] is not None:
                k = torch.cat([cache[0], k], 2); v = torch.cat([cache[1], v], 2)
            if self.window and k.shape[2] > self.window:
                k, v = k[:, :, -self.window:], v[:, :, -self.window:]
            new_cache = (k, v)
        if self.sink_k is not None:
            sk = self.sink_k.expand(B, -1, -1, -1).to(k.dtype)
            k = torch.cat([sk, k], 2)
            v = torch.cat([torch.zeros_like(sk), v], 2)      # zero value -> denominator only
        # The attention math itself stays BF16: arXiv:2509.25149 keeps attention out of FP4.
        kw = {"enable_gqa": True} if self.nkv != self.nh else {}
        if cache is not None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False, **kw)
        else:
            mask = masks[1 if self.is_global else 0]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **kw)
        out = out.transpose(1, 2).reshape(B, L, self.nh * self.hd)
        out = self.o_proj(out)
        return (out, new_cache) if cache is not None else out


class SwiGLU(nn.Module):
    def __init__(self, idx: int, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or D_FF
        L = linear_factory("ffn", idx)
        self.gate_proj, self.up_proj = L(cfg.d_model, d_ff), L(cfg.d_model, d_ff)
        self.down_proj = L(d_ff, cfg.d_model)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFFN(nn.Module):
    # Top-k routed experts + an always-on shared expert, with DeepSeek-V3's aux-loss-free load
    # balancing: no auxiliary loss term, just a per-expert bias nudged toward even utilisation.
    def __init__(self, idx: int):
        super().__init__()
        self.experts = nn.ModuleList(SwiGLU(idx) for _ in range(cfg.moe_experts))
        self.shared = nn.ModuleList(SwiGLU(idx) for _ in range(cfg.moe_shared))
        self.router = nn.Linear(cfg.d_model, cfg.moe_experts, bias=False, dtype=BF16)
        self.register_buffer("bias", torch.zeros(cfg.moe_experts, dtype=torch.float32))
        self.register_buffer("load", torch.zeros(cfg.moe_experts, dtype=torch.float32))

    def forward(self, x):
        B, L, D = x.shape
        flat = x.reshape(-1, D)
        logits = self.router(flat).float()
        gate = logits.sigmoid()
        top = (gate + self.bias).topk(cfg.moe_top_k, dim=-1).indices
        w = gate.gather(-1, top)
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-6)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            hit = (top == e)
            if not hit.any():
                continue
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            wr = (w * hit.float()).sum(-1)[rows].unsqueeze(-1).to(flat.dtype)
            out.index_add_(0, rows, expert(flat[rows]) * wr)
            if self.training:
                self.load[e] += rows.numel()
        for sh in self.shared:
            out = out + sh(flat)
        return out.view(B, L, D)

    @torch.no_grad()
    def rebalance(self):
        if not cfg.moe or self.load.sum() == 0:
            return
        share = self.load / self.load.sum()
        self.bias -= cfg.moe_bias_speed * (share - 1.0 / cfg.moe_experts).sign()
        self.load.zero_()


class Block(nn.Module):
    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx
        self.pre_attn, self.post_attn = RMSNorm(cfg.d_model, cfg.rms_eps), RMSNorm(cfg.d_model, cfg.rms_eps)
        self.pre_ffn, self.post_ffn = RMSNorm(cfg.d_model, cfg.rms_eps), RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mixer = self._make_mixer(idx)
        self.ffn = MoEFFN(idx) if cfg.moe else SwiGLU(idx)
        # tag the residual-output projections so _init can depth-scale their std
        if isinstance(self.mixer, Attention):
            self.mixer.o_proj._is_residual_out = True
        for m in self.ffn.modules():
            if isinstance(m, SwiGLU):
                m.down_proj._is_residual_out = True

    def _make_mixer(self, idx):
        if cfg.hybrid_linear and HAS_FLA and (idx + 1) % cfg.global_every != 0:
            from fla.layers import GatedDeltaNet          # linear-attention slot
            return GatedDeltaNet(hidden_size=cfg.d_model, num_heads=cfg.n_heads,
                                 expand_v=1, use_gate=True).to(BF16)
        return Attention(idx)

    def forward(self, x, rope, masks, cache=None):
        h = self.pre_attn(x)
        new_cache = cache
        if isinstance(self.mixer, Attention):
            if cache is not None:
                h, new_cache = self.mixer(h, rope, masks, cache)
            else:
                h = self.mixer(h, rope, masks)
        else:
            out = self.mixer(h)                      # gated DeltaNet returns (o, ...) in fla
            h = out[0] if isinstance(out, tuple) else out
        x = x + self.post_attn(h)
        x = x + self.post_ffn(self.ffn(self.pre_ffn(x)))
        return (x, new_cache) if cache is not None else x


class AletheiaChat(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE_ACTUAL, cfg.d_model, dtype=BF16)
        self.blocks = nn.ModuleList(Block(i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, VOCAB_SIZE_ACTUAL, bias=False, dtype=BF16)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        # Multi-token prediction (DeepSeek-V3): extra blocks predicting t+2, t+3, ... They share
        # the LM head, are trained with weight `mtp_weight`, and are dropped at export unless you
        # want speculative decoding.
        self.mtp = nn.ModuleList(Block(cfg.n_layers - 1) for _ in range(cfg.mtp_depth))
        self.mtp_norm = nn.ModuleList(RMSNorm(cfg.d_model, cfg.rms_eps) for _ in range(cfg.mtp_depth))
        self._rope = None
        self.apply(self._init)

    def _init(self, m):
        # depth-scaled init: residual branches shrink with sqrt(2*n_layers) (GPT-NeoX/OLMo style)
        if isinstance(m, nn.Linear) or (TE is not None and isinstance(m, TE.Linear)):
            std = (cfg.d_model ** -0.5)
            if getattr(m, "_is_residual_out", False):
                std /= math.sqrt(2 * cfg.n_layers)
            nn.init.trunc_normal_(m.weight, std=std, a=-3 * std, b=3 * std)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02, a=-0.06, b=0.06)

    def rope(self, device):
        if self._rope is None:
            n = max(cfg.seq_len, cfg.window) * 4
            self._rope = (rope_cache(n, cfg.head_dim, cfg.rope_local, device),
                          rope_cache(n, cfg.head_dim, cfg.rope_global, device))
        return self._rope

    def hidden(self, idx, want_pre_norm: bool = False):
        rope = self.rope(idx.device)
        masks = build_masks(idx, idx.device)
        h = self.embed(idx)
        for b in self.blocks:
            if cfg.act_ckpt and self.training:
                h = torch.utils.checkpoint.checkpoint(b, h, rope, masks, use_reentrant=False)
            else:
                h = b(h, rope, masks)
        return (self.norm(h), h, rope, masks) if want_pre_norm else self.norm(h)

    def _ce(self, h, targets, loss_mask):
        logits = self.lm_head(h).float()             # logits always FP32 for the softmax
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                             reduction="none")
        if loss_mask is not None:
            m = loss_mask.reshape(-1).float()
            ce = (ce * m).sum() / m.sum().clamp(min=1)
        else:
            ce = ce.mean()
        if cfg.z_loss:
            lse = torch.logsumexp(logits.view(-1, logits.size(-1)), dim=-1)
            ce = ce + cfg.z_loss * lse.pow(2).mean()
        return ce

    def forward(self, idx, targets=None, loss_mask=None):
        if targets is None:
            return self.lm_head(self.hidden(idx)).float()
        h, pre, rope, masks = self.hidden(idx, want_pre_norm=True)
        loss = self._ce(h, targets, loss_mask)
        # --- multi-token prediction: head d predicts token t+1+d ------------
        for d, (blk, nrm) in enumerate(zip(self.mtp, self.mtp_norm), start=1):
            if idx.shape[1] <= d:
                break
            hp = blk(pre, rope, masks)
            tgt = targets[:, d:]
            lm = None if loss_mask is None else loss_mask[:, d:]
            loss = loss + cfg.mtp_weight * self._ce(nrm(hp)[:, :-d], tgt, lm)
        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=256, temperature=0.7, top_p=0.9,
                 rep_penalty=1.05, stop_ids=(EOS_ID,)):
        caches = [(None, None) for _ in self.blocks]
        rope = self.rope(idx.device)
        out = idx
        cur = idx
        for _ in range(max_new_tokens):
            h = self.embed(cur)
            new = []
            for b, c in zip(self.blocks, caches):
                h, nc = b(h, rope, None, cache=c)
                new.append(nc)
            caches = new
            logits = self.lm_head(self.norm(h))[:, -1].float()
            if rep_penalty != 1.0:
                for t in set(out[0].tolist()):
                    logits[0, t] /= rep_penalty
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = logits.softmax(-1)
                sp, si = probs.sort(-1, descending=True)
                keep = (sp.cumsum(-1) - sp) < top_p
                sp = torch.where(keep, sp, torch.zeros_like(sp))
                sp = sp / sp.sum(-1, keepdim=True)
                nxt = si.gather(-1, torch.multinomial(sp, 1))
            out = torch.cat([out, nxt], 1)
            cur = nxt
            if nxt.item() in stop_ids:
                break
        return out


torch.manual_seed(cfg.seed)
model = AletheiaChat().to("cuda" if HAS_CUDA else "cpu")
n_total = sum(p.numel() for p in model.parameters())
n_emb = model.embed.weight.numel() * (1 if cfg.tie_embeddings else 2)
print(f"[=] {cfg.name}: {n_total/1e9:.3f}B params ({(n_total-n_emb)/1e9:.3f}B non-embedding)")
print(f"    layers={cfg.n_layers} d={cfg.d_model} d_ff={D_FF} heads={cfg.n_heads}/{cfg.n_kv_heads}"
      f" head_dim={cfg.head_dim} window={cfg.window} global_every={cfg.global_every}")
q = [i for i in range(cfg.n_layers)
     if cfg.bf16_first_blocks <= i < cfg.n_layers - cfg.bf16_last_blocks]
print(f"    {cfg.precision.upper()} linears in blocks {q[0] if q else '-'}..{q[-1] if q else '-'}"
      f"  ({len(q)}/{cfg.n_layers}); BF16: first {cfg.bf16_first_blocks} + last {cfg.bf16_last_blocks},"
      f" embeddings, lm_head, norms, attention")
""")

# ---------------------------------------------------------------- 6. recipe
md(r"""
## 6. The NVFP4 recipe

`NVFP4BlockScaling` defaults *are* the paper's recipe: 16×16 random Hadamard transform on the
column-wise (Wgrad) operands, stochastic rounding on gradients only, 2D 16×16 weight scaling,
E4M3 per-16 block scales under an FP32 global scale. The flags below therefore mostly *disable*
things, and each is exposed only so the ablation table later in the notebook can be produced.

If your Wgrad run destabilises (the AMD/PSU MXFP4 finding, arXiv:2605.09825 — Wgrad is the primary
divergence driver, and *deterministic* Hadamard rotation fixes what stochastic rounding cannot), set
`cfg.deterministic_hadamard=True`, which drops the random sign vector via
`NVTE_NVFP4_DISABLE_RHT`-adjacent env control and falls back to a fixed rotation.
""")

code(r"""
def make_recipe(precision: str):
    if TE is None or precision == "bf16":
        return None
    if precision == "nvfp4":
        if cfg.deterministic_hadamard:
            os.environ["NVTE_NVFP4_DETERMINISTIC_RHT"] = "1"   # honoured by TE >= 2.19 builds
        return NVFP4BlockScaling(
            disable_rht=not cfg.nvfp4_rht,
            disable_stochastic_rounding=not cfg.nvfp4_stochastic_rounding,
            disable_2d_quantization=not cfg.nvfp4_2d_weights,
            fp8_dpa=cfg.fp4_attention,
            fp8_mha=cfg.fp4_attention,
        )
    if precision == "mxfp8":
        return MXFP8BlockScaling()
    if precision == "fp8":
        return Float8CurrentScaling()
    raise ValueError(precision)


recipe = make_recipe(cfg.precision)
print("recipe:", type(recipe).__name__ if recipe else "None (pure BF16)")
if recipe is not None:
    for f in ("disable_rht", "disable_stochastic_rounding", "disable_2d_quantization",
              "fp4_format", "fp8_format", "fp8_dpa"):
        if hasattr(recipe, f):
            print(f"  {f:32s} = {getattr(recipe, f)}")

# TE renamed fp8_autocast -> autocast; support both.
if TE is not None:
    try:
        from transformer_engine.pytorch.quantization import autocast as te_autocast
        def quant_ctx(enabled, rcp):
            return te_autocast(enabled=enabled, recipe=rcp)
    except Exception:
        def quant_ctx(enabled, rcp):
            return TE.fp8_autocast(enabled=enabled, fp8_recipe=rcp)
else:
    from contextlib import nullcontext
    def quant_ctx(enabled, rcp):
        return nullcontext()

print("quantized autocast:", quant_ctx(recipe is not None, recipe).__class__.__name__)
""")

# ---------------------------------------------------------------- 7. optimizer
md(r"""
## 7. Optimizer — Muon on the matrices, AdamW on everything else

Muon orthogonalises the momentum-augmented gradient via Newton–Schulz and applies only to 2D hidden
matrices; biases, embeddings, norms and the LM head go to AdamW. NVIDIA's *SOAP, Muon, and Beyond*
(arXiv:2607.20548) finds Muon and SOAP beating AdamW at multi-billion × multi-trillion scale with
update-RMS-matched learning rates, and Muon holds data efficiency far past AdamW's critical batch
size (arXiv:2505.02222), which is exactly the regime a 13k tok/param budget forces you into.

Schedule: **WSD** (warmup → stable → `1−sqrt` decay). Unlike cosine, WSD needs no advance commitment
to the total step count — you can extend a run — and it interacts cleanly with the NVFP4→BF16 switch:
put the switch at the start of the decay phase so the last 18% of tokens are both high-precision and
low-LR, which is where arXiv:2509.25149 recovers most of the 1.5% → 0.5% loss gap.
""")

code(r"""
def split_params(m):
    hidden2d, other = [], []
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        is_emb = "embed" in name or "lm_head" in name
        (hidden2d if (p.ndim == 2 and not is_emb) else other).append(p)
    return hidden2d, other


hidden2d, other = split_params(model)
print(f"Muon params: {sum(p.numel() for p in hidden2d)/1e9:.3f}B in {len(hidden2d)} tensors")
print(f"AdamW params: {sum(p.numel() for p in other)/1e6:.1f}M in {len(other)} tensors")

MuonCls = getattr(torch.optim, "Muon", None)
if MuonCls is None:
    class MuonCls(torch.optim.Optimizer):
        # Minimal Newton-Schulz Muon (Jordan et al.); used when torch.optim.Muon is absent (<2.9).
        def __init__(self, params, lr=2e-2, momentum=0.95, nesterov=True, ns_steps=5,
                     weight_decay=0.0):
            super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                          ns_steps=ns_steps, weight_decay=weight_decay))

        @staticmethod
        def _zeropower(G, steps):
            a, b, c = 3.4445, -4.7750, 2.0315
            X = G.bfloat16()
            X = X / (X.norm() + 1e-7)
            transposed = X.size(0) > X.size(1)
            if transposed:
                X = X.T
            for _ in range(steps):
                A = X @ X.T
                B = b * A + c * A @ A
                X = a * X + B @ X
            return (X.T if transposed else X)

        @torch.no_grad()
        def step(self, closure=None):
            for g in self.param_groups:
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    buf = st.setdefault("m", torch.zeros_like(p))
                    buf.mul_(g["momentum"]).add_(p.grad)
                    upd = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                    upd = self._zeropower(upd, g["ns_steps"])
                    # RMS-match the update to AdamW's ~1.0 scale (Jordan's 0.2*sqrt(max(dim)))
                    upd = upd * 0.2 * math.sqrt(max(p.shape))
                    if g["weight_decay"]:
                        p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.add_(upd.to(p.dtype), alpha=-g["lr"])
    print("[!] torch.optim.Muon unavailable -> bundled implementation")

# ---- FP32 master weights -----------------------------------------------------
# Compute runs in BF16/NVFP4; the optimizer owns an FP32 copy. Without this, a bf16 master weight
# has 2**-8 relative resolution, so late-training updates below half the representable spacing
# round to zero and the embedding path silently stops learning through the whole decay phase.
class MasterWeights:
    def __init__(self, params, enabled=True):
        self.enabled = enabled
        self.pairs = [(p, p.detach().float().clone()) for p in params] if enabled else []
        for _, m in self.pairs:
            m.requires_grad_(True)

    def to_master(self):
        for p, m in self.pairs:
            m.grad = p.grad.float() if p.grad is not None else None

    def to_model(self):
        for p, m in self.pairs:
            p.data.copy_(m.data.to(p.dtype))

    @property
    def master_params(self):
        return [m for _, m in self.pairs]


class EMA:
    # Weight EMA: a cheap, uncontroversial final-checkpoint quality gain. Off by default because
    # it costs one FP32 copy of the model.
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().float().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point} if decay > 0 else {}

    @torch.no_grad()
    def update(self, model):
        if not self.shadow:
            return
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)


master_hidden = MasterWeights(hidden2d, cfg.fp32_master)
master_other = MasterWeights(other, cfg.fp32_master)
MASTERS = [master_hidden, master_other]

opt_muon = MuonCls(master_hidden.master_params if cfg.fp32_master else hidden2d,
                   lr=cfg.lr_muon, momentum=0.95, ns_steps=5, weight_decay=cfg.weight_decay)
opt_adamw = torch.optim.AdamW(master_other.master_params if cfg.fp32_master else other,
                              lr=cfg.lr_adamw, betas=(0.9, 0.95), eps=1e-8,
                              weight_decay=0.0, fused=HAS_CUDA and not cfg.fp32_master)
OPTS = [opt_muon, opt_adamw]
BASE_LRS = [cfg.lr_muon, cfg.lr_adamw]
print(f"master weights: {'fp32' if cfg.fp32_master else 'bf16 (in-place)'}")

DECAY_START = WARMUP + int(TOTAL_STEPS * cfg.hold_frac)
BF16_SWITCH_STEP = int(TOTAL_STEPS * cfg.bf16_switch_frac)

def lr_scale(step: int) -> float:
    # Warmup - Hold - Cosine decay. The hold does the learning; the decay can be re-cut later
    # without having wasted the middle of the run, which matters when the budget is a guess.
    if step < WARMUP:
        return step / WARMUP
    if step < DECAY_START:
        return 1.0
    prog = (step - DECAY_START) / max(1, TOTAL_STEPS - DECAY_START)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

def set_lr(step: int):
    s = lr_scale(step)
    for opt, base in zip(OPTS, BASE_LRS):
        for g in opt.param_groups:
            g["lr"] = cfg.lr_min + (base - cfg.lr_min) * s
    return s

print(f"WHD: warmup {WARMUP:,} | hold -> {DECAY_START:,} | cosine -> {TOTAL_STEPS:,} (floor {cfg.lr_min:g})")
print(f"NVFP4 -> BF16 switch at step {BF16_SWITCH_STEP:,} ({cfg.bf16_switch_frac:.0%} of tokens)")

ema = EMA(model, cfg.ema_decay)
""")

# ---------------------------------------------------------------- 8. train
md(r"""
## 8. Training loop

Gradient accumulation, `torch.compile`, activation checkpointing, per-step telemetry, a checkpoint
manifest (ported from `aletheiaMLX.ipynb` and extended with optimizer + RNG + config), periodic
validation, and the precision switch. The quantized autocast wraps **only the forward pass** — TE
records the quantization scheme there and reuses it in backward.
""")

code(r"""
import time
from contextlib import nullcontext

DEV = "cuda" if HAS_CUDA else "cpu"
CKPT = cfg.root / "ckpt"

fwd = model
if cfg.compile and hasattr(torch, "compile"):
    try:
        fwd = torch.compile(model, dynamic=False)
        print("[=] torch.compile enabled")
    except Exception as exc:
        print(f"[!] compile disabled: {exc}")


class CheckpointManager:
    def __init__(self, root: Path, keep: int = 3):
        self.root, self.keep = root, keep
        self.manifest = root / "manifest.json"

    def save(self, step, telemetry):
        d = self.root / f"step_{step:07d}"; d.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "muon": opt_muon.state_dict(), "adamw": opt_adamw.state_dict(),
                    "step": step, "cfg": {k: str(v) for k, v in asdict(cfg).items()},
                    "rng": torch.get_rng_state()}, d / "state.pt")
        (d / "telemetry.json").write_text(json.dumps(telemetry))
        self.manifest.write_text(json.dumps({"step": step, "path": str(d)}))
        old = sorted(self.root.glob("step_*"))[:-self.keep]
        for o in old:
            shutil.rmtree(o, ignore_errors=True)
        print(f"[ckpt] step {step}")

    def load(self):
        blank = {"step": [], "loss": [], "val": [], "lr": [], "gnorm": [], "tok": 0, "precision": []}
        if not self.manifest.exists():
            return 0, blank
        m = json.loads(self.manifest.read_text())
        st = torch.load(Path(m["path"]) / "state.pt", map_location=DEV, weights_only=False)
        model.load_state_dict(st["model"])
        opt_muon.load_state_dict(st["muon"]); opt_adamw.load_state_dict(st["adamw"])
        tel = json.loads((Path(m["path"]) / "telemetry.json").read_text())
        print(f"[ckpt] resumed at step {st['step']}")
        return st["step"], tel


ckpt = CheckpointManager(CKPT)
start_step, telemetry = ckpt.load()


@torch.no_grad()
def evaluate(n_batches: int = 20) -> float:
    model.eval(); tot = 0.0
    it = iter(val_loader)
    for i in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        with quant_ctx(False, None):
            tot += model(x.to(DEV), y.to(DEV)).item()
    model.train()
    return tot / max(1, i + 1)


def train(total_steps: int = TOTAL_STEPS, start: int = None):
    from tqdm.auto import tqdm
    step0 = start_step if start is None else start
    model.train()
    bar = tqdm(total=total_steps, initial=step0, desc=f"{cfg.name} [{cfg.precision}]",
               dynamic_ncols=True)
    t0, session_tok = time.time(), 0
    for step in range(step0 + 1, total_steps + 1):
        # --- precision switch: last (1-bf16_switch_frac) of tokens in BF16
        use_q = recipe is not None and step < BF16_SWITCH_STEP
        if step == BF16_SWITCH_STEP and recipe is not None:
            print(f"\n[precision] step {step}: {cfg.precision.upper()} -> BF16 for the tail")
        scale = set_lr(step)
        cur_lr = opt_muon.param_groups[0]["lr"]
        for o in OPTS:
            o.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(cfg.grad_accum):
            x, y = next(train_iter)
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with quant_ctx(use_q, recipe):
                loss = fwd(x, y) / cfg.grad_accum
            loss.backward()
            loss_acc += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        for m in MASTERS:
            m.to_master()
        for o in OPTS:
            o.step()
        for m in MASTERS:
            m.to_model()
        if cfg.moe:
            for blk in model.blocks:
                if isinstance(blk.ffn, MoEFFN):
                    blk.ffn.rebalance()
        ema.update(model)

        tok = TOKENS_PER_STEP
        telemetry["tok"] += tok; session_tok += tok
        telemetry["step"].append(step); telemetry["loss"].append(loss_acc)
        telemetry["lr"].append(cur_lr); telemetry["gnorm"].append(gnorm)
        telemetry["precision"].append(cfg.precision if use_q else "bf16")

        if step % cfg.eval_every == 0:
            v = evaluate(); telemetry["val"].append([step, v])
        el = max(time.time() - t0, 1e-6)
        bar.set_postfix(loss=f"{loss_acc:.4f}", ppl=f"{math.exp(min(loss_acc,20)):.1f}",
                        lr=f"{cur_lr:.2e}", gn=f"{gnorm:.2f}",
                        tok_s=f"{session_tok/el:,.0f}", prec="q" if use_q else "bf16")
        bar.update(1)
        if step % cfg.save_every == 0 or step == total_steps:
            ckpt.save(step, telemetry)
    bar.close()
    return telemetry


telemetry = train()
""")

code(r"""
# ---- 8b. telemetry plots ----------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

def plot(t, ema=0.95):
    if not t["step"]:
        print("[!] nothing logged yet"); return
    s = np.array(t["step"]); l = np.array(t["loss"])
    sm, acc = [], None
    for v in l:
        acc = v if acc is None else ema * acc + (1 - ema) * v
        sm.append(acc)
    sm = np.array(sm)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    ax[0].plot(s, l, alpha=.2, color="tab:blue"); ax[0].plot(s, sm, color="tab:blue", lw=2)
    if t["val"]:
        v = np.array(t["val"]); ax[0].plot(v[:, 0], v[:, 1], "o-", color="tab:red", label="val")
        ax[0].legend()
    if BF16_SWITCH_STEP <= s.max():
        for a in ax[:2]:
            a.axvline(BF16_SWITCH_STEP, ls="--", c="k", alpha=.5)
    ax[0].set(title="cross-entropy", xlabel="step", ylabel="loss"); ax[0].grid(alpha=.3)
    ax[1].plot(s, np.exp(np.minimum(sm, 20)), color="tab:orange", lw=2)
    ax[1].set(title="perplexity (EMA)", xlabel="step", yscale="log"); ax[1].grid(alpha=.3)
    ax[2].plot(s, t["gnorm"], color="tab:green", alpha=.7, label="grad norm")
    ax2 = ax[2].twinx(); ax2.plot(s, t["lr"], color="tab:purple", label="lr")
    ax[2].set(title="grad norm (green) / lr (purple)", xlabel="step"); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.show()
    print(f"tokens seen: {t['tok']/1e9:.3f}B  ({t['tok']/max(1,NONEMB_PARAMS):.0f} tok/param)")

plot(telemetry)
""")

# ---------------------------------------------------------------- 9. benchmarks
md(r"""
## 9. Benchmarks — precision, throughput, memory, quality

Four independent axes, each measurable in this notebook:

1. **Precision fidelity** — the only number that decides whether NVFP4 was worth it: the relative
   validation-loss gap versus a BF16 twin on identical data and seed. The paper's bar is **<1.5%**
   during stable training, **~0.5%** after the BF16 tail.
2. **Throughput / MFU** — tokens/s and model-FLOPs-utilisation per recipe. FP4 buys 2–3× arithmetic
   over FP8 on paper; realised gains in published runs are ~9–10% end-to-end over FP8
   (arXiv:2605.09825) up to ~20% (nanochat TE thread), because non-GEMM work does not shrink.
3. **Memory** — peak allocated bytes per recipe.
4. **Quality** — lm-eval-harness v2 hard suite plus an Aletheia-specific knowledge probe, because no
   public benchmark measures "does it know what a Context Fabric is".
""")

code(r"""
# ---- 9a. recipe A/B: loss gap, throughput, memory ---------------------------
import copy, gc

def short_run(precision: str, steps: int = 30, seed: int = 0):
    torch.manual_seed(seed)
    m = AletheiaChat().to(DEV)
    h2, oth = split_params(m)
    om = MuonCls(h2, lr=cfg.lr_muon, momentum=0.95, ns_steps=5, weight_decay=cfg.weight_decay)
    oa = torch.optim.AdamW(oth, lr=cfg.lr_adamw, betas=(0.9, 0.95), fused=HAS_CUDA)
    rcp = make_recipe(precision)
    loader = PackedShards("train", cfg.seq_len, cfg.micro_batch, seed=123)
    it = infinite(loader)
    if HAS_CUDA:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    losses, t0, ntok = [], time.time(), 0
    for _ in range(steps):
        for o in (om, oa):
            o.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(cfg.grad_accum):
            x, y = next(it)
            with quant_ctx(rcp is not None, rcp):
                l = m(x.to(DEV), y.to(DEV)) / cfg.grad_accum
            l.backward(); acc += l.item()
            ntok += x.numel()
        torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.grad_clip)
        for o in (om, oa):
            o.step()
        losses.append(acc)
    if HAS_CUDA:
        torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if HAS_CUDA else 0.0
    out = {"precision": precision, "final_loss": sum(losses[-5:]) / 5,
           "tok_s": ntok / dt, "peak_gb": peak, "losses": losses}
    del m, om, oa; gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
    return out


candidates = ["bf16"] + [p for p in ("fp8", "mxfp8", "nvfp4")
                         if {"fp8": FP8_OK, "mxfp8": MXFP8_OK, "nvfp4": NVFP4_OK}[p]]
results = {}
for p in candidates:
    try:
        results[p] = short_run(p, steps=20)
        print(f"[{p:6s}] loss={results[p]['final_loss']:.4f} "
              f"tok/s={results[p]['tok_s']:,.0f} peak={results[p]['peak_gb']:.2f} GB")
    except Exception as exc:
        print(f"[{p:6s}] FAILED: {type(exc).__name__}: {exc}")

if "bf16" in results:
    base = results["bf16"]
    print(f"\n{'recipe':8s}{'loss':>10s}{'rel gap':>10s}{'tok/s':>12s}{'speedup':>10s}{'peak GB':>10s}")
    for p, r in results.items():
        gap = (r["final_loss"] - base["final_loss"]) / base["final_loss"] * 100
        print(f"{p:8s}{r['final_loss']:10.4f}{gap:9.2f}%{r['tok_s']:12,.0f}"
              f"{r['tok_s']/base['tok_s']:9.2f}x{r['peak_gb']:10.2f}")
    print("\nbar from arXiv:2509.25149: |rel gap| < 1.5% mid-training, ~0.5% after the BF16 tail")
""")

code(r"""
# ---- 9b. MFU + inference latency / KV-cache footprint -----------------------
def model_flops_per_token(c: Cfg, d_ff: int, seq: int) -> float:
    # 6*N for fwd+bwd on the matmuls, plus attention terms; sliding window shrinks the quadratic part
    n_attn = c.d_model * (c.n_heads + 2 * c.n_kv_heads) * c.head_dim + c.n_heads * c.head_dim * c.d_model
    n_ffn = 3 * c.d_model * d_ff
    dense = 6 * c.n_layers * (n_attn + n_ffn)
    globals_ = c.n_layers / c.global_every
    locals_ = c.n_layers - globals_
    attn_flops = 12 * (globals_ * seq + locals_ * min(seq, c.window)) * c.head_dim * c.n_heads
    return dense + attn_flops

PEAK = {  # dense BF16 TFLOP/s, vendor spec; FP4 is ~2x BF16 dense on Blackwell
    "bf16": {"B200": 2250, "RTX PRO 6000": 503, "RTX 5090": 419, "RTX 5070": 123},
}
def guess_peak():
    for k, v in PEAK["bf16"].items():
        if k.split()[-1] in GPU:
            return v * 1e12
    return None

fpt = model_flops_per_token(cfg, D_FF, cfg.seq_len)
print(f"model FLOPs/token (fwd+bwd): {fpt/1e9:.2f} GFLOP")
for p, r in results.items():
    pk = guess_peak()
    mfu = fpt * r["tok_s"] / pk * 100 if pk else float("nan")
    print(f"  {p:6s} {r['tok_s']:>10,.0f} tok/s   MFU {mfu:5.1f}%" if pk else
          f"  {p:6s} {r['tok_s']:>10,.0f} tok/s")

kv_bytes = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * 2   # k+v, bf16
eff = sum(min(cfg.window, 1 << 20) if (i + 1) % cfg.global_every else (1 << 20)
          for i in range(cfg.n_layers)) / cfg.n_layers
print(f"\nKV cache: {kv_bytes/1024:.1f} KB/token dense; sliding window caps the local layers at "
      f"{cfg.window} -> ~{kv_bytes*eff/(1<<20)/1024:.2f} GB at 1M ctx"
      f" vs {kv_bytes*(1<<20)/1e9:.2f} GB if every layer were global")

if HAS_CUDA:
    model.eval()
    prompt = torch.tensor([[BOS_ID] + tokenizer.encode("<|user|>hi<|eot|><|assistant|>",
                                                       add_special_tokens=False).ids], device=DEV)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        _ = model.generate(prompt, max_new_tokens=64, temperature=0.0)
    torch.cuda.synchronize()
    print(f"decode: {64/(time.time()-t0):.1f} tok/s (batch 1, greedy, no fused kernels)")
    model.train()
""")

md(r"""
### 9c. Ablation grid (optional, expensive)

Every knob the papers disagree about, in one table. Run this only when you have spare GPU-hours; on
the `laptop` preset each row is a 20-step run.

| axis | values | source of the disagreement |
|---|---|---|
| RHT | on / off | arXiv:2509.25149 (random, Wgrad only) vs arXiv:2605.09825 (deterministic) |
| stochastic rounding | on / off | 2509.25149 says required on grads; 2605.09825 says insufficient alone |
| 2D weight scaling | on / off | 2509.25149 |
| BF16 block count | 0 / 2+4 / 2+8 | 2509.25149 keeps ~16% |
| FP4 attention | off / on | arXiv:2607.04422 full-stack FP4 |
| optimizer | Muon / AdamW | arXiv:2607.20548 |
| superwords | SuperBPE / stage-1 BPE | arXiv:2503.13423 |
""")

code(r"""
ABLATIONS = {
    "nvfp4 default":        dict(),
    "no RHT":               dict(nvfp4_rht=False),
    "no stochastic round":  dict(nvfp4_stochastic_rounding=False),
    "no 2D weight scale":   dict(nvfp4_2d_weights=False),
    "no BF16 blocks":       dict(bf16_first_blocks=0, bf16_last_blocks=0),
    "FP4 attention":        dict(fp4_attention=True),
    "deterministic RHT":    dict(deterministic_hadamard=True),
}

RUN_ABLATIONS = False        # flip to True when you have the budget
if RUN_ABLATIONS and NVFP4_OK:
    rows = []
    for label, patch in ABLATIONS.items():
        saved = {k: getattr(cfg, k) for k in patch}
        for k, v in patch.items():
            setattr(cfg, k, v)
        try:
            r = short_run("nvfp4", steps=20)
            rows.append((label, r["final_loss"], r["tok_s"], r["peak_gb"]))
        except Exception as exc:
            rows.append((label, float("nan"), float("nan"), float("nan")))
            print(f"[{label}] FAILED {type(exc).__name__}: {exc}")
        for k, v in saved.items():
            setattr(cfg, k, v)
    print(f"\n{'ablation':24s}{'loss':>10s}{'tok/s':>12s}{'peak GB':>10s}")
    for label, l, t, m in rows:
        print(f"{label:24s}{l:10.4f}{t:12,.0f}{m:10.2f}")
else:
    print("ablation grid skipped (set RUN_ABLATIONS=True)")
""")

# ---------------------------------------------------------------- 10. anneal
md(r"""
## 10. Anneal into a chat model that knows Aletheia OS

The SmolLM3 recipe's shape: the last slice of pretraining is a *mid-training* phase on high-value,
instruction-shaped data at decaying LR — not a separate SFT run. Three sources, at the same time as
the NVFP4 → BF16 switch:

1. **SmolTalk2 (SFT split)** — general chat/instruction behaviour.
2. **Synthetic Aletheia QA** — generated *mechanically* from the repo: every ADR, PRD/SAD section and
   public Rust item becomes question/answer pairs anchored in real file content. No teacher model
   required, so the facts are the repo's facts.
3. **Aletheia pipeline traces** — templated `<|intent|> → <|context|> → <|plan|> → <|capability|> →
   <|policy|> → <|action|> → <|provenance|>` transcripts, so the model learns the OS's actual control
   flow as a conversational form.

Loss is masked to assistant spans only.
""")

code(r"""
# ---- 10a. build the anneal set ---------------------------------------------
ANNEAL = cfg.root / "corpus" / "anneal.jsonl"

def rust_items(text: str):
    pat = re.compile(r"^\s*(?:pub\s+)?(fn|struct|enum|trait|impl|mod)\s+([A-Za-z_][A-Za-z0-9_]*)",
                     re.M)
    return [(m.group(1), m.group(2), m.start()) for m in pat.finditer(text)]

def synth_aletheia_qa(limit: int = 20000):
    repo = clone_aletheia(); out = []
    for p in iter_repo_files(repo):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = p.relative_to(repo)
        if p.suffix in DOC_EXT:
            # split on markdown headings -> (question from heading, answer from body)
            parts = re.split(r"^(#{1,4})\s+(.+)$", txt, flags=re.M)
            for i in range(1, len(parts) - 2, 3):
                head, body = parts[i + 1].strip(), parts[i + 2].strip()
                if len(body) < 120:
                    continue
                out.append({"messages": [
                    {"role": "system", "content": "You are Aletheia, the assistant of the Aletheia "
                                                  "operating system. Answer from the system's own "
                                                  "documentation and cite the file."},
                    {"role": "user", "content": f"In Aletheia OS, {head.rstrip('?.')}?"},
                    {"role": "assistant", "content": f"{body[:2000]}\n\n(source: {rel})"},
                ]})
        elif p.suffix == ".rs":
            for kind, name, off in rust_items(txt)[:12]:
                snippet = txt[off: off + 1200]
                out.append({"messages": [
                    {"role": "user", "content": f"What does the `{name}` {kind} do in Aletheia, and "
                                                f"where does it live?"},
                    {"role": "assistant", "content":
                        f"`{name}` is a `{kind}` defined in `{rel}`:\n\n<|code|>```rust\n{snippet}\n```"},
                ]})
        if len(out) >= limit:
            break
    print(f"[+] synthetic Aletheia QA: {len(out)}")
    return out

PIPELINE_TEMPLATE = (
    "<|intent|>{intent}\n"
    "<|context|>{context}\n"
    "<|plan|>{plan}\n"
    "<|capability|>{cap}\n"
    "<|policy|>{policy}\n"
    "<|action|>{action}\n"
    "<|provenance|>{prov}"
)

def pipeline_traces(n: int = 4000):
    verbs = [("read", "fs.read"), ("write", "fs.write"), ("spawn", "proc.spawn"),
             ("open a socket", "net.connect"), ("mount", "fs.mount"),
             ("load a WASM component", "component.instantiate")]
    rng = random.Random(cfg.seed); out = []
    for i in range(n):
        verb, cap = rng.choice(verbs)
        target = rng.choice(["/etc/policy.toml", "/dev/nvme0", "/srv/index.db",
                             "component://hello", "10.0.0.4:8443"])
        approved = rng.random() > 0.3
        body = PIPELINE_TEMPLATE.format(
            intent=f"user asks to {verb} {target}",
            context=f"Context Fabric returned 3 capability-scoped records for {target}",
            plan=f"1. resolve {cap}\n2. attenuate to {target}\n3. request policy approval\n4. execute",
            cap=f"{cap}(target={target}) attenuated from the caller's authority",
            policy="approved by human operator" if approved else
                   "DENIED: no standing approval for this authority class",
            action=f"executed {cap} deterministically" if approved else "aborted before execution",
            prov=f"recorded intent->action chain #{i:06d}, verified against the plan",
        )
        out.append({"messages": [
            {"role": "user", "content": f"{verb.capitalize()} {target}."},
            {"role": "assistant", "content": body},
        ]})
    print(f"[+] pipeline traces: {len(out)}")
    return out


def build_anneal(target_tokens: int):
    rows = synth_aletheia_qa() + pipeline_traces()
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceTB/smoltalk2", "SFT", split="train", streaming=True)
        got = 0
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                rows.append({"messages": msgs}); got += 1
            if got >= len(rows):
                break
        print(f"[+] smoltalk2: {got}")
    except Exception as exc:
        print(f"[!] smoltalk2 unavailable ({type(exc).__name__}) - Aletheia-only anneal")
    random.Random(cfg.seed).shuffle(rows)
    with open(ANNEAL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[=] anneal set: {len(rows)} conversations -> {ANNEAL}")
    return ANNEAL

build_anneal(int(0.02 * TOKEN_BUDGET))
""")

code(r"""
# ---- 10b. masked anneal loop (assistant spans only) -------------------------
ASSIST_ID = tokenizer.token_to_id("<|assistant|>")
EOT_ID = tokenizer.token_to_id("<|eot|>")

def encode_conversation(msgs, max_len: int):
    text = hf_tok.apply_chat_template(msgs, tokenize=False)
    ids = tokenizer.encode(text, add_special_tokens=False).ids[:max_len]
    mask, on = [0] * len(ids), False
    for i, t in enumerate(ids):
        if t == ASSIST_ID:
            on = True; mask[i] = 0; continue
        if t == EOT_ID and on:
            mask[i] = 1; on = False; continue
        mask[i] = 1 if on else 0
    return ids, mask


def anneal_batches(path: Path, batch: int, max_len: int):
    buf_x, buf_m = [], []
    while True:
        with open(path, encoding="utf-8") as f:
            for line in f:
                ids, m = encode_conversation(json.loads(line)["messages"], max_len)
                if sum(m) == 0:
                    continue
                pad = max_len - len(ids)
                buf_x.append(ids + [PAD_ID] * pad); buf_m.append(m + [0] * pad)
                if len(buf_x) == batch:
                    x = torch.tensor(buf_x); m = torch.tensor(buf_m)
                    yield x[:, :-1], x[:, 1:], m[:, 1:]
                    buf_x, buf_m = [], []


def anneal(steps: int = 500, lr_frac: float = 0.1):
    from tqdm.auto import tqdm
    it = anneal_batches(ANNEAL, cfg.micro_batch, cfg.seq_len)
    for opt, base in zip(OPTS, BASE_LRS):
        for g in opt.param_groups:
            g["lr"] = base * lr_frac
    model.train()
    bar = tqdm(total=steps, desc="anneal (BF16, masked)", dynamic_ncols=True)
    for step in range(steps):
        for o in OPTS:
            o.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(cfg.grad_accum):
            x, y, m = next(it)
            with quant_ctx(False, None):          # anneal in BF16 by design
                l = model(x.to(DEV), y.to(DEV), loss_mask=m.to(DEV)) / cfg.grad_accum
            l.backward(); acc += l.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for o in OPTS:
            o.step()
        bar.set_postfix(loss=f"{acc:.4f}"); bar.update(1)
    bar.close()
    ckpt.save(TOTAL_STEPS + steps, telemetry)

anneal(steps=200)
""")

# ---------------------------------------------------------------- 11. eval
md(r"""
## 11. Quality evaluation

Two halves. Public benchmarks tell you whether the model is a competent small LM; only the Aletheia
probe tells you whether it learned *this operating system*. The v2 hard suite (MMLU-Pro, GPQA, BBH,
MuSR, IFEval, MATH-L5) is used because the v1 set is saturated — but note honestly that at ~1.5B and
a small token budget most hard-suite numbers sit near chance; the informative signals at this scale
are validation perplexity, HellaSwag/ARC/PIQA/WinoGrande, and the domain probe.
""")

code(r"""
# ---- 11a. Aletheia knowledge probe (closed-book, exact + fuzzy) -------------
PROBE = [
    ("What is Aletheia?", ["operating system", "AI-native", "from scratch", "not derived from Linux"]),
    ("Which language is the Aletheia kernel written in?", ["rust", "no_std"]),
    ("Which CPU architectures does Aletheia target?", ["x86", "amd64", "risc-v", "riscv", "arm"]),
    ("What are the domain primitives?",
     ["entity", "capability", "context", "intent", "action", "memory", "relationship"]),
    ("What property do Aletheia capabilities have?",
     ["unforgeable", "attenuat", "revocable"]),
    ("How does the policy engine relate to the capability engine?",
     ["independent", "human approval", "governance", "separate"]),
    ("What does Aletheia use instead of RAG?", ["context fabric", "capability-aware", "structured retrieval"]),
    ("How are components isolated?", ["wasm", "no ambient authority", "component runtime"]),
    ("What is the deterministic pipeline?",
     ["intent", "context", "proposal", "validat", "capability", "policy", "execut", "provenance"]),
    ("Is the AI subsystem trusted?", ["untrusted", "collaborator", "proposes", "validates"]),
]

@torch.no_grad()
def probe(max_new_tokens: int = 160, temperature: float = 0.2):
    model.eval(); total = 0.0
    for q, keys in PROBE:
        text = hf_tok.apply_chat_template(
            [{"role": "system", "content": "You are Aletheia. Answer factually and briefly."},
             {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False).ids], device=DEV)
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature,
                             stop_ids=(EOS_ID, EOT_ID))
        ans = tokenizer.decode(out[0].tolist()[ids.shape[1]:]).lower()
        hit = sum(k.lower() in ans for k in keys) / len(keys)
        total += hit
        print(f"[{hit:4.0%}] {q}\n       -> {ans[:220].strip()}\n")
    model.train()
    print(f"Aletheia knowledge score: {total/len(PROBE):.1%}")
    return total / len(PROBE)

probe()
""")

code(r"""
# ---- 11b. public benchmarks via lm-evaluation-harness -----------------------
HF_EXPORT = cfg.root / "hf_export"

def export_hf():
    # Minimal HF-compatible export so lm-eval / transformers can load the model.
    HF_EXPORT.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in model.state_dict().items()}
    if cfg.tie_embeddings:
        sd.pop("lm_head.weight", None)      # shares storage with embed.weight; safetensors refuses
                                            # aliases, and `tie_word_embeddings` restores it on load
    save_file(sd, str(HF_EXPORT / "model.safetensors"), metadata={"format": "pt"})
    (HF_EXPORT / "config.json").write_text(json.dumps({
        "model_type": "aletheia_chat", "architectures": ["AletheiaChatForCausalLM"],
        "vocab_size": VOCAB_SIZE_ACTUAL, "hidden_size": cfg.d_model, "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads, "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim, "intermediate_size": D_FF, "rms_norm_eps": cfg.rms_eps,
        "sliding_window": cfg.window, "global_attn_every_n_layers": cfg.global_every,
        "rope_theta_local": cfg.rope_local, "rope_theta_global": cfg.rope_global,
        "qk_norm": cfg.qk_norm, "tie_word_embeddings": cfg.tie_embeddings,
        "torch_dtype": "bfloat16", "bos_token_id": BOS_ID, "eos_token_id": EOS_ID,
        "pad_token_id": PAD_ID, "trained_precision": cfg.precision,
    }, indent=2))
    hf_tok.save_pretrained(HF_EXPORT)
    print(f"[=] exported to {HF_EXPORT}")

export_hf()

EASY = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "openbookqa",
        "lambada_openai", "mmlu"]
HARD = ["mmlu_pro", "gpqa_diamond_zeroshot", "bbh", "musr", "ifeval", "minerva_math"]
CODE = ["humaneval", "mbpp"]

print("suggested lm-eval invocation (needs an lm-eval model adapter for this architecture,\n"
      "or convert the export to a supported architecture first):\n")
print(f"  lm_eval --model hf --model_args pretrained={HF_EXPORT},trust_remote_code=True,dtype=bfloat16 \\\n"
      f"          --tasks {','.join(EASY)} --batch_size auto --num_fewshot 0\n")
print(f"  lm_eval --model hf --model_args pretrained={HF_EXPORT},trust_remote_code=True,dtype=bfloat16 \\\n"
      f"          --tasks {','.join(HARD)} --batch_size auto\n")

if HAS_LMEVAL:
    print("lm-eval is installed. At ~1.5B with a small token budget expect near-chance on the hard\n"
          "suite; read validation PPL, the easy suite, and the Aletheia probe instead.")
""")

code(r"""
# ---- 11c. held-out perplexity by domain ------------------------------------
@torch.no_grad()
def ppl_by_domain(samples: dict):
    model.eval(); rows = []
    for name, text in samples.items():
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) < 8:
            continue
        x = torch.tensor([ids[:-1]], device=DEV); y = torch.tensor([ids[1:]], device=DEV)
        with quant_ctx(False, None):
            loss = model(x, y).item()
        rows.append((name, loss, math.exp(min(loss, 20)), len(ids)))
    model.train()
    print(f"{'domain':12s}{'loss':>9s}{'ppl':>10s}{'tokens':>9s}")
    for n, l, p, t in rows:
        print(f"{n:12s}{l:9.4f}{p:10.2f}{t:9d}")
    return rows

ppl_by_domain(SAMPLES)
""")

# ---------------------------------------------------------------- 12. chat
md(r"""
## 12. Talk to it
""")

code(r"""
def chat(user: str, system: str = "You are Aletheia, the assistant of the Aletheia operating system.",
         history=None, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9):
    msgs = [{"role": "system", "content": system}] + (history or []) + \
           [{"role": "user", "content": user}]
    text = hf_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False).ids], device=DEV)
    model.eval()
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_p=top_p, stop_ids=(EOS_ID, EOT_ID))
    model.train()
    reply = tokenizer.decode(out[0].tolist()[ids.shape[1]:])
    return reply.replace("<|eot|>", "").strip()

for q in ["What is the Context Fabric and why is it not RAG?",
          "Write a Rust function that attenuates a capability before delegation.",
          "Walk me through what happens when I ask you to mount a filesystem."]:
    print(f"\n### {q}\n{chat(q)}")
""")

# ---------------------------------------------------------------- 13. scale
md(r"""
## 13. Scaling this run up

What to change, in order, to go from the `laptop` smoke run to a real 13k-tok/param model:

1. `PRESET = "cluster"` — 2048×30, seq 4096, micro-batch 8×24 accumulation.
2. **Distribute.** Wrap in FSDP2 (`torch.distributed.fsdp.fully_shard`) and set
   `NVFP4BlockScaling`'s amax/quantization group to the data-parallel group by passing
   `amax_reduction_group` to the quantized autocast. TE's NVFP4 supports distributed training; the
   scale-reduction group must match your DP group or the global FP32 scale drifts per rank.
3. **Data.** The `mix` quotas are fractions of the token budget; at trillions of tokens stream
   directly rather than materialising shards, or pre-shard with a Rust/`datatrove` job.
4. **Sequence length.** Train the bulk at 4096 and extend context in the anneal phase by raising
   `rope_global` (the local layers keep 10k) — Gemma 4's split-base scheme is designed for exactly
   this.
5. **Precision tail.** Keep `bf16_switch_frac = 0.82`. It is the cheapest 1 point of relative loss
   available.
6. **Watch Wgrad.** If loss spikes appear only after FP4 is enabled on gradients, the MXFP4 study
   says the culprit is Wgrad, not your LR: set `deterministic_hadamard=True` before touching the
   schedule.
7. **Muon at scale.** Muon's advantage grows with batch size past AdamW's critical batch — raise
   global batch before raising LR.

### Deliberate non-choices

* **MoE** — better quality-per-FLOP, worse quality-per-byte; an OS assistant is memory-bound on
  device.
* **Full linear-attention hybrid by default** — layer selection dominates the outcome
  (arXiv:2606.30562); the 3:1 sliding-window:global split gets most of the KV saving with none of the
  recall risk. Flag exists.
* **FP4 optimizer states / attention** — arXiv:2607.04422 shows it works; it also adds two failure
  modes to a run that already has one novel number format. Flag exists, default off.
* **Distillation from a large teacher** — the strongest single quality lever at 1.5B, and orthogonal
  to everything here. It needs a teacher and a licence decision, so it is out of scope for a
  pretraining notebook.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parents[1] / "Aletheia_NVFP4_Pretrain.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  ({len(cells)} cells, {out.stat().st_size/1024:.0f} KB)")
