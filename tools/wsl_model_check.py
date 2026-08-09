"""Build the real model on the GPU and push one batch of random ids through it.

Not a training run: no dataset, no optimizer, no checkpoint, no tokenizer. It answers the
questions that only appear once every piece is assembled -- do the gated-DeltaNet slots line up
with the attention slots, does the FP4 autocast wrap a forward that contains both, does the MTP
head's loss reach the parameters, and does Muon get the tensors it is supposed to get.

    /opt/ale/bin/python tools/wsl_model_check.py
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

NB = Path(__file__).resolve().parent.parent / "Aletheia_NVFP4_Pretrain.ipynb"
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]
src = lambda i: "".join(cells[i]["source"])                              # noqa: E731

ns: dict = {}
exec(compile(src(2), "cell_probe", "exec"), ns)                          # capability probe
exec(compile(src(4), "cell_config", "exec"), ns)                         # the real config
cfg = ns["cfg"]

# Escape hatch for the prebuilt-TE limitation: NVIDIA's cu13 core wheel carries sm_120 but not
# sm_120a, and the FP4 stochastic-rounding cast is emitted only for the latter. ALE_NO_SR=1
# turns SR off so the rest of the FP4 path can be exercised on a stock install.
import os
if os.environ.get("ALE_NO_SR") == "1":
    cfg.nvfp4_stochastic_rounding = False
    print("[!] stochastic rounding disabled (ALE_NO_SR=1)")

# The tokenizer cells are skipped -- substitute what they would have produced.
ns.update(VOCAB_SIZE_ACTUAL=cfg.vocab_size, EOS_ID=2, math=math)
exec(compile(src(16), "cell_arch", "exec"), ns)                          # architecture + model
exec(compile(src(18), "cell_recipe", "exec"), ns)                        # precision recipe

model, recipe, quant_ctx = ns["model"], ns["recipe"], ns["quant_ctx"]
Attention = ns["Attention"]

kinds = ["attn" if isinstance(b.mixer, Attention) else "deltanet" for b in model.blocks]
print(f"\nmixers      : {kinds.count('deltanet')} gated-DeltaNet + {kinds.count('attn')} attention")
print(f"              {''.join('A' if k == 'attn' else 'd' for k in kinds)}")
glob = [i for i, b in enumerate(model.blocks) if isinstance(b.mixer, Attention) and b.mixer.is_global]
print(f"global attn : layers {glob}  (every {cfg.global_every})")

hidden2d = [p for n, p in model.named_parameters()
            if p.ndim == 2 and "embed" not in n and "lm_head" not in n]
other = [p for n, p in model.named_parameters()
         if not (p.ndim == 2 and "embed" not in n and "lm_head" not in n)]
print(f"Muon        : {sum(p.numel() for p in hidden2d)/1e6:.1f}M in {len(hidden2d)} tensors")
print(f"AdamW       : {sum(p.numel() for p in other)/1e6:.1f}M in {len(other)} tensors")

dev = "cuda"
# Default to a small shape (integration check). Pass the real micro-batch and sequence length to
# measure the peak VRAM a training step will actually take:  wsl_model_check.py 2 2048
import sys
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
L = int(sys.argv[2]) if len(sys.argv) > 2 else 256
idx = torch.randint(4, cfg.vocab_size, (B, L + 1), device=dev)
idx[:, L // 3] = ns["EOS_ID"]                                            # a document boundary
x, y = idx[:, :-1], idx[:, 1:]

torch.cuda.reset_peak_memory_stats()
model.train()
with quant_ctx(recipe is not None, recipe):
    loss = model(x, y)
loss.backward()

missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
print(f"\nloss        : {loss.item():.4f}   (ln(vocab) = {math.log(cfg.vocab_size):.4f} at init)")
print(f"precision   : {cfg.precision}   recipe {type(recipe).__name__ if recipe else 'None'}")
print(f"peak VRAM   : {torch.cuda.max_memory_allocated()/2**30:.2f} GB   (B={B}, L={L})")
print(f"params without a gradient: {len(missing)}" + (f" -> {missing[:5]}" if missing else ""))

ok = torch.isfinite(loss).item() and not missing
print(f"\n{'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
