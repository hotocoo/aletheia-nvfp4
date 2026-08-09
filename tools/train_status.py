"""Print the state of a pretraining run from its checkpoint artifacts.

tqdm scrollback in a log file is not a progress signal you can query; the checkpoint manifest and
the telemetry written beside it are. Both are refreshed every `cfg.save_every` steps.

    python tools/train_status.py                 # ./aletheia_nvfp4
    python tools/train_status.py _smoke          # a smoke-run root
"""

import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "aletheia_nvfp4")
manifest = root / "ckpt" / "manifest.json"

if not manifest.exists():
    shards = root / "shards" / "train_meta.json"
    if shards.exists():
        print(f"no checkpoint yet -- shards are built ({json.loads(shards.read_text())['tokens']:,} "
              f"tokens); the run is between shard build and step {1}")
    else:
        print(f"no checkpoint and no shards under {root} -- the run is still building its corpus")
    raise SystemExit(0)

m = json.loads(manifest.read_text())
tel = json.loads((Path(m["path"]) / "telemetry.json").read_text())
step = m["step"]
age = time.time() - manifest.stat().st_mtime

loss = tel["loss"][-1] if tel["loss"] else float("nan")
window = tel["loss"][-50:] or [loss]
val = tel["val"][-1] if tel["val"] else None

print(f"step        : {step:,}")
print(f"tokens seen : {tel['tok']:,}")
print(f"loss        : {loss:.4f}   (mean of last {len(window)}: {sum(window)/len(window):.4f})")
if val:
    print(f"val loss    : {val[1]:.4f}  @ step {val[0]:,}")
print(f"lr / gnorm  : {tel['lr'][-1]:.3e} / {tel['gnorm'][-1]:.2f}")
print(f"precision   : {tel['precision'][-1]}")
print(f"checkpoint  : {m['path']}  ({age/60:.1f} min old)")
if age > 3600:
    print("            ^ stale by more than an hour -- check whether the run is still alive")
