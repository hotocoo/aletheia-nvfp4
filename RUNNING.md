# Running the pretrain yourself

Everything here is copy-pasteable from **PowerShell** on the Windows host. The run lives inside
WSL2; you never need to open a WSL shell unless you want to.

Throughout, `$REPO` is the repo as WSL sees it:

```powershell
$REPO = "/mnt/c/Users/9700X-5070/Downloads/github/aletheia-nvfp4"
$PY   = "/opt/ale/bin/python"
```

---

## 0. Before anything: cap WSL2's memory

`C:\Users\9700X-5070\.wslconfig` — already in place, shown here so you know why it matters:

```ini
[wsl2]
memory=12GB
processors=8
swap=8GB
```

Without a cap, WSL2 takes about half the host RAM and never gives the Linux page cache back.
That plus anything running natively is what froze this machine. If you edit it, apply with
`wsl --shutdown` (which kills any running training — check first).

---

## 1. One-time setup

Run these in order. Each is idempotent; re-running a completed step is a no-op.

```powershell
wsl -d Ubuntu -u root -- bash $REPO/tools/wsl_setup.sh        # torch cu130, deps, fla
wsl -d Ubuntu -u root -- bash $REPO/tools/wsl_cuda_te.sh      # CUDA toolkit 13.3 (~3 GB download)
wsl -d Ubuntu -u root -- bash $REPO/tools/wsl_te_source.sh    # Transformer Engine, sm_120a
```

Two things about these that cost a lot of time to discover:

**The toolkit must be 13.3, not 13.0.** Ubuntu 26.04 ships glibc 2.43, which declares `rsqrt` and
`rsqrtf`; CUDA 13.0 declares them incompatibly, so its `nvcc` cannot compile *any* `.cu` file on
this system. It surfaces as `CMAKE_CXX_COMPILER not set, after EnableLanguage`, which points
nowhere near the real cause. `wsl_cuda_te.sh` verifies the toolkit with a trivial compile before
you build anything against it.

**Transformer Engine must be built from source** — 30-90 minutes, the long pole. NVIDIA's prebuilt
wheel carries `sm_120` but not `sm_120a`, and the FP4 conversion PTX only exists for the latter, so
a stock install reports `NVFP4: True` and then aborts inside the first quantized GEMM. The failing
kernels are in TE's *core* library, so rebuilding only the PyTorch extension does not help.

**Do not run anything heavy alongside it.** Four `nvcc` jobs against a 12 GB cap is already most
of what the VM has.

---

## 2. Verify before you commit 14 days

```powershell
wsl -d Ubuntu -u root -- $PY $REPO/tools/wsl_probe.py
```

This is the gate. It must print:

```
NVFP4       : True
```

`False` means the run will fall back to BF16 and take far longer for a worse result. Stop and fix
it rather than starting.

You can confirm the build carries the right architecture directly:

```powershell
wsl -d Ubuntu -u root -- bash $REPO/tools/wsl_te_archs.sh
```

```
=== libtransformer_engine.so
    archs: sm_120a sm_90a
    -> FP4 stochastic rounding: supported
```

**One deviation from the published recipe is unavoidable on this GPU.** FP4 *stochastic rounding*
is a datacenter-Blackwell instruction — TE gates it on `sm_100a`/`sm_103a` (B200/B300) and consumer
`sm_120a` is not in that set, at any build setting. The config detects `sm_120`/`sm_121` and turns
`nvfp4_stochastic_rounding` off automatically, printing:

```
[!] sm_120: FP4 stochastic rounding unavailable on consumer Blackwell -> disabled
```

Everything else in arXiv:2509.25149's recipe runs: the random Hadamard transform on Wgrad inputs,
2D 16×16 weight scaling, BF16 for the first 2 and last 4 blocks, and the BF16 tail for the final
18% of tokens. Expect a slightly wider FP4-vs-BF16 loss gap than the paper's <1.5%, since some of
that margin is attributed to SR. Left enabled, the run still completes but prints a per-thread
device error on every quantized backward pass and the gradient cast silently falls back.

Two more checks, both seconds-to-minutes, neither trains anything:

```powershell
wsl -d Ubuntu -u root -- $PY $REPO/tools/attn_parity.py          # FlexAttention == fallback == decode
wsl -d Ubuntu -u root -- $PY $REPO/tools/wsl_model_check.py 2 2048   # builds the real model, one fwd/bwd
```

Expected from the model check on a 12 GB 5070:

```
mixers      : 18 gated-DeltaNet + 6 attention
peak VRAM   : 4.77 GB   (B=2, L=2048)
params without a gradient: 0
PASS
```

Add roughly 3 GB for optimizer state during real training (FP32 masters 1.45, Muon momentum 1.32,
AdamW 0.27) → about **7.8 GB peak of your 12**. If you ever see an OOM, drop `micro_batch` to 1 and
raise `grad_accum` to 16: identical tokens per step, half the activation memory.

---

## 3. Start the run

```powershell
wsl -d Ubuntu -u root -- bash $REPO/tools/wsl_pretrain.sh
```

What it does: copies the notebook to `/root/aletheia-run` (ext4 — shard reads over `/mnt/c` go
through the 9p bridge and cost more than the GPU does), picks up your Hugging Face token from the
Windows-side cache if WSL has none, and runs the whole notebook under papermill with live output.

To keep it running while you use the machine for other things, launch it detached:

```powershell
Start-Process powershell -ArgumentList '-NoProfile','-Command',
  "wsl -d Ubuntu -u root -- bash $REPO/tools/wsl_pretrain.sh" -WindowStyle Hidden
```

**The first training step takes several minutes with no output.** FlexAttention compiles its
kernels for your exact batch shape, `torch.compile` traces the model, and TE initializes its FP4
kernels — all before step 1 prints. This is not a hang. Wait at least 10 minutes before
concluding otherwise.

---

## 4. Watching it

### Progress — the reliable way

```powershell
wsl -d Ubuntu -u root -- $PY $REPO/tools/train_status.py /root/aletheia-run/aletheia_nvfp4
```

```
step        : 4,200
tokens seen : 17,203,200
loss        : 3.8412   (mean of last 50: 3.8677)
val loss    : 3.9102  @ step 4,000
lr / gnorm  : 1.87e-03 / 0.41
precision   : nvfp4
checkpoint  : /root/aletheia-run/aletheia_nvfp4/ckpt/step_0004200  (1.3 min old)
```

Read from the checkpoint manifest and telemetry, both rewritten every 200 steps. It also warns you
when the checkpoint is more than an hour stale, which is how you find out a run died without
watching a log.

Poll it on a loop:

```powershell
while ($true) { cls; wsl -d Ubuntu -u root -- $PY $REPO/tools/train_status.py /root/aletheia-run/aletheia_nvfp4; sleep 60 }
```

### Verbose — the raw log

```powershell
wsl -d Ubuntu -u root -- tail -f /root/aletheia-run/logs/pretrain.log
```

Everything papermill sees: per-cell markers, the shard builder's source resolution, the tqdm bar
with live `loss / ppl / lr / gnorm / tok_s / prec`. `prec` flips from `q` to `bf16` at 82% of
tokens — that switch is intentional, it's the recipe's BF16 tail.

Last 100 lines instead of following:

```powershell
wsl -d Ubuntu -u root -- tail -100 /root/aletheia-run/logs/pretrain.log
```

### GPU

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 5
```

Healthy mid-training: utilization 90%+, memory ~8 GB. Utilization repeatedly dropping to near zero
means the data loader is starving the GPU, not that the GPU is slow.

### Host memory

```powershell
wsl -d Ubuntu -u root -- free -h
```

Should sit well under the 12 GB cap. If `available` approaches zero, stop the run — that is the
state that froze the machine before.

---

## 5. Stages and what they cost

The notebook runs top to bottom. On an RTX 5070:

| stage | what happens | time |
|---|---|---|
| probe | prints your precision path — must say NVFP4 | seconds |
| config | params, step count, honest budget reality check | seconds |
| tokenizer | trains both SuperBPE stages over ~1.5 GB | 1-2 h, cached after |
| data | streams the mixture into `uint16` shards | hours, resumable |
| model / recipe / optimizer | builds it all | seconds |
| **train** | the run itself, checkpointing every 200 steps | `train_days`, default 14 |
| benchmarks | NVFP4-vs-BF16 loss gap, tok/s, MFU, KV cache | ~20 min |
| anneal | masked chat mid-training on assistant spans | ~1 h |
| eval | lm-eval, per-domain perplexity, Aletheia probe | ~20 min |
| chat | talk to it | — |

The tokenizer and shard stages both cache. A second run skips straight to training.

---

## 6. Stopping and resuming

Stop it however you like — Ctrl-C, closing the window, a reboot. Checkpoints land every 200 steps
and carry model weights, both optimizer states, the FP32 master weights, the EMA shadow and the RNG
state, so a resume continues rather than restarts.

To resume, run the same launch command. It reads `ckpt/manifest.json` and prints:

```
[ckpt] resumed at step 4200
```

Only the last 3 checkpoints are kept. To keep one permanently, copy it out of `ckpt/` first.

---

## 7. Changing the run

Edit **`tools/build_notebook.py`**, not the notebook — the notebook is generated and your edits
would be overwritten. Then:

```powershell
python tools/build_notebook.py
```

Knobs worth knowing, all in the config cell:

| knob | effect |
|---|---|
| `train_days` | wall-clock budget; step count is derived from measured throughput |
| `PRESET` | `rtx5070` / `workstation` / `cluster` — changes model shape |
| `micro_batch`, `grad_accum` | VRAM vs. steps; their product sets tokens/step |
| `precision` | force `bf16` to compare against the FP4 path |
| `hybrid_linear` | `False` reverts DeltaNet slots to sliding-window attention |
| `save_every`, `eval_every` | checkpoint and validation frequency |

---

## 8. When something looks wrong

| symptom | cause | fix |
|---|---|---|
| `NVFP4: False` in the probe | TE built without `sm_120a`, or prebuilt wheel installed | rerun `wsl_te_source.sh`; check with `wsl_te_archs.sh` |
| `FP4 cvt PTX instructions are architecture-specific` | stochastic rounding on a consumer GPU | expected — the config disables SR on `sm_120`; if you forced it on, turn it back off |
| `CMAKE_CXX_COMPILER not set, after EnableLanguage` | CUDA 13.0 vs glibc 2.43 | use CUDA 13.3; verify with `wsl_nvcc_probe.sh /usr/local/cuda-13.3` |
| `Could not find transformer-engine PyPI package` | prebuilt wheel metadata alongside a source build | `wsl_te_source.sh` now uninstalls the wheels first — rerun it |
| `PackageNotFoundError: transformer_engine_cu13` | probe asked for the wheel's name after a source build | fixed in `wsl_probe.py`; pull the current version |
| `libnccl.so.2: cannot open shared object file` | TE imported before torch | import torch first; the notebook already does |
| No output for 10+ min at step 1 | kernel compilation | normal, wait |
| `A different number of tensors was saved...` | checkpointing outside the FP4 autocast | already fixed; means you're on an old notebook — rebuild it |
| `lm-eval: False` in the probe | benchmark suite not installed | `/opt/ale/bin/python -m pip install lm-eval`; the eval stage skips silently without it |
| OOM | activation memory | `micro_batch=1`, `grad_accum=16` |
| GPU utilization near zero | data loader starving | check the log for shard-building or HF download retries |
| Whole machine freezes | WSL2 uncapped, or two heavy jobs at once | check `.wslconfig`; never build and train simultaneously |
| Checkpoint stale >1 h | run died | `tail` the log for the traceback, then relaunch — it resumes |
