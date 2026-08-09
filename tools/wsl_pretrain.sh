#!/usr/bin/env bash
# Launch the pretraining run inside WSL2 on the FP4 path.
#
#   wsl -d Ubuntu -u root -- bash /mnt/c/.../tools/wsl_pretrain.sh
#
# Artifacts (corpus, tokenizer, shards, checkpoints) are written to WORK on the ext4 filesystem,
# NOT to /mnt/c: shard reads go through the 9p bridge there and cost more than the GPU does.
# Interrupt at any time -- the training cell resumes from ckpt/manifest.json on the next run.
set -eu

REPO=/mnt/c/Users/9700X-5070/Downloads/github/aletheia-nvfp4
WORK=/root/aletheia-run
VENV=/opt/ale

mkdir -p "$WORK"
cp "$REPO/Aletheia_NVFP4_Pretrain.ipynb" "$WORK/"

# Hugging Face token: read from the Windows-side cache if WSL has none of its own.
if [ -z "${HF_TOKEN:-}" ] && [ -f /mnt/c/Users/9700X-5070/.cache/huggingface/token ]; then
    HF_TOKEN=$(cat /mnt/c/Users/9700X-5070/.cache/huggingface/token)
    export HF_TOKEN
fi

cd "$WORK"
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "run root : $WORK"
echo "log      : $WORK/logs/pretrain.log"
echo "status   : $VENV/bin/python $REPO/tools/train_status.py $WORK/aletheia_nvfp4"
echo

# PIPESTATUS, not $?, or you record tee's exit status instead of papermill's. A run that ends
# without this line in the log was killed outright rather than failing -- the distinction matters,
# because a kill points at the platform (WSL session teardown, OOM) and a non-zero exit points at
# the notebook.
set +e
"$VENV/bin/python" -m papermill \
    Aletheia_NVFP4_Pretrain.ipynb logs/pretrain_out.ipynb \
    --log-output --no-progress-bar 2>&1 | tee -a logs/pretrain.log
rc=${PIPESTATUS[0]}
printf '[exit] papermill rc=%s at %s\n' "$rc" "$(date -Is)" | tee -a logs/pretrain.log
exit "$rc"
