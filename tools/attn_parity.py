"""Check that the FlexAttention path and the manual fallback compute the same function.

The two paths exist because FlexAttention lowers through Triton and native Windows has no
backend for it. They express the sink differently -- Flex folds the sink logit into the
denominator afterwards via the log-sum-exp the kernel returns, the manual path adds it to the
denominator directly -- so "they agree" is an assumption that has to be tested, not assumed.

Runs the architecture cell straight out of the notebook at toy size, so it tests the shipped
code rather than a copy of it. Attention only: no data, no optimizer, no training.

    python tools/attn_parity.py
"""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

NB = Path(__file__).resolve().parent.parent / "Aletheia_NVFP4_Pretrain.ipynb"
cells = [c for c in json.loads(NB.read_text(encoding="utf-8"))["cells"]]
src = lambda i: "".join(cells[i]["source"])                              # noqa: E731

PROBE_CELL, ARCH_CELL = 2, 16
assert "capability probe" in src(PROBE_CELL)
assert "class Attention" in src(ARCH_CELL)

ns: dict = {}
exec(compile(src(PROBE_CELL), "cell_probe", "exec"), ns)

# Toy config: small enough to compare exhaustively, large enough that GQA, the sliding window,
# partial RoPE and the document mask are all exercised.
ns["cfg"] = SimpleNamespace(
    name="parity", precision="bf16", seed=0,
    d_model=256, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=64,
    rms_eps=1e-5, qk_norm=True, tie_embeddings=True,
    global_every=2, window=8, rope_local=1e4, rope_global=1e6, partial_rope=0.25,
    attn_sinks=True, intra_doc_mask=True, hybrid_linear=False,
    mtp_depth=0, mtp_weight=0.0, z_loss=0.0, act_ckpt=False, seq_len=32,
    moe=False, moe_experts=2, moe_top_k=1, moe_shared=1, moe_bias_speed=1e-3,
    bf16_first_blocks=0, bf16_last_blocks=0,
)
ns.update(D_FF=256, VOCAB_SIZE_ACTUAL=256, EOS_ID=2, HAS_FLA=False, math=math)
exec(compile(src(ARCH_CELL), "cell_arch", "exec"), ns)

if not ns["USE_FLEX"]:
    print("FlexAttention unavailable here -- nothing to compare against. "
          "Run this inside WSL2 on the GPU.")
    sys.exit(0)

torch.manual_seed(0)
dev = "cuda"
model, cfg = ns["model"], ns["cfg"]
B, L = 2, 32
idx = torch.randint(4, 256, (B, L), device=dev)
idx[:, L // 2] = ns["EOS_ID"]                       # a document boundary inside the pack
x = torch.randn(B, L, cfg.d_model, device=dev, dtype=torch.bfloat16)
rope = model.rope(dev)

worst = 0.0
for name, layer in (("window", 0), ("global", 1)):
    mixer = model.blocks[layer].mixer
    assert mixer.is_global == (name == "global"), "layer/global mapping changed"

    ns["USE_FLEX"] = True
    with torch.no_grad():
        flex = mixer(x, rope, ns["build_masks"](idx, dev))
    ns["USE_FLEX"] = False
    with torch.no_grad():
        manual = mixer(x, rope, ns["build_masks"](idx, dev))

    err = (flex.float() - manual.float()).abs().max().item()
    scale = manual.float().abs().max().item()
    worst = max(worst, err)
    print(f"{name:7s} layer: max abs diff {err:.5f}  (output scale {scale:.3f})")

# Decode parity: one token against a cache must match the same token computed in a full pass.
# No document boundary here -- the decode path has no notion of one, so a packed-sequence mask
# would make the two disagree by design rather than by bug.
ns["USE_FLEX"] = True
mixer = model.blocks[1].mixer                        # global layer: no window truncation
plain = torch.full_like(idx, 5)
with torch.no_grad():
    full = mixer(x, rope, ns["build_masks"](plain, dev))
    cache = (None, None)
    for t in range(L):                               # feed one token at a time
        step, cache = mixer(x[:, t:t + 1], rope, None, cache=cache)
step_err = (step.float() - full[:, -1:].float()).abs().max().item()
print(f"decode  step  : max abs diff {step_err:.5f}  (vs the same position in a full pass)")

TOL = 3e-2                                           # BF16 accumulation over 32 positions
worst = max(worst, step_err)
print(f"\n{'PASS' if worst <= TOL else 'FAIL'}: worst {worst:.5f} (tolerance {TOL})")
sys.exit(0 if worst <= TOL else 1)
