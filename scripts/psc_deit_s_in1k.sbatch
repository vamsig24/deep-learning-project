#!/bin/bash
# DeiT-S + distillation token on ImageNet-1k.
# PSC details follow scripts/train_vit_small_8_v100.sbatch from
# https://github.com/lukashann95/ViT-perforated
# One full GPU node, 8x V100-32. One parent; torchrun starts the 8 processes.
# 128 per GPU and --grad-accum-steps 1 keeps the global batch at 1024 (lr 1e-3).
#
#SBATCH --job-name=deit-s-in1k
#SBATCH --account=cis260364p
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:8
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
JOB_START=$(date +%s)

# Clone this repo to $HOME/IDl_project on PSC, or set REPO to wherever it lives.
# The job runs train.py in that directory, not a browser download named train(1).py.
REPO="${REPO:-$HOME/IDl_project}"
TRAIN_PY="$REPO/train.py"
IMAGENET=/ocean/datasets/community/imagenet
WORK_DIR="/ocean/projects/cis260364p/${USER}/deit-s-in1k-${SLURM_JOB_ID}"

# Same launcher as the ViT-Small job. There is no conda env name to activate.
command -v uv

# Avoid CPU thread oversubscription across the eight training processes.
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Once, on a login node, so eight ranks do not download the teacher together:
#   uv run python -c "import timm; timm.create_model('regnety_160.deit_in1k', pretrained=True)"
# Optional: export WANDB_API_KEY before sbatch. train.py reads it from the environment.

# Fail before staging data if required paths or the training file are missing.
: "${LOCAL:?PSC node-local storage path is not set}"
test -r "$IMAGENET/train.tar"
test -r "$IMAGENET/val.tar"
test -f "$TRAIN_PY"
echo "Training file: $TRAIN_PY"

# Stage archives, then extract locally. The archives already contain train/ and val/.
# $LOCAL is wiped between jobs, so every submit copies and extracts again.
mkdir -p "$LOCAL/imagenet"

echo "Copying ImageNet..."
copy_started=$SECONDS
for split in train val; do
  echo "Copying ImageNet $split.tar..."
  split_started=$SECONDS
  rsync -a --info=progress2 --stats --outbuf=L \
    "$IMAGENET/$split.tar" "$LOCAL/imagenet/"
  echo "ImageNet $split archive copy complete in $((SECONDS - split_started)) seconds."
  echo "Extracting ImageNet $split..."
  extract_started=$SECONDS
  tar --extract --file="$LOCAL/imagenet/$split.tar" \
    --directory="$LOCAL/imagenet" --no-same-owner --no-same-permissions \
    --checkpoint=100000 --checkpoint-action=echo
  test -d "$LOCAL/imagenet/$split"
  echo "ImageNet $split extraction complete in $((SECONDS - extract_started)) seconds."
  rm -- "$LOCAL/imagenet/$split.tar"
done
echo "ImageNet staging complete in $((SECONDS - copy_started)) seconds total."

cd "$REPO"
mkdir -p "$WORK_DIR"

# Staging comes out of the 24 h. Python stops itself before Slurm kills the job.
USED_H=$(awk -v s="$JOB_START" -v n="$(date +%s)" 'BEGIN{printf "%.2f",(n-s)/3600}')
REMAIN_H=$(awk -v u="$USED_H" 'BEGIN{printf "%.2f", 23.3-u}')
echo "Setup used ${USED_H}h. Python budget ${REMAIN_H}h."
awk -v r="$REMAIN_H" 'BEGIN{exit !(r+0 > 0)}' || {
  echo "No time left for training."
  exit 1
}

# 40 CPUs / 8 GPUs is five dataloader workers per rank. train.py does that split.
uv run torchrun --standalone --nproc_per_node=8 "$TRAIN_PY" \
  --data "$LOCAL/imagenet" \
  --work-dir "$WORK_DIR" \
  --batch-size 128 \
  --grad-accum-steps 1 \
  --max-runtime-hours "$REMAIN_H"

# Not submitted from here. On PSC, after train.py is in $HOME/IDl_project:
#   sbatch psc_deit_s_in1k.sbatch
# Smoke first:
#   uv run torchrun --standalone --nproc_per_node=8 "$HOME/IDl_project/train.py" \
#     --data "$LOCAL/imagenet" --work-dir "$WORK_DIR" --grad-accum-steps 1 --smoke
