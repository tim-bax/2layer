#!/bin/bash
#SBATCH -J SHD_1layer_JAX
#SBATCH -t 16:00:00
#SBATCH -o out/SHD_1layer_JAX_%j.out

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.bax@uos.de

echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ---- Disable user-site packages ----
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# ---- Project root (parent of 1layer_version; data/ and 1layer_version/ live here) ----
PROJECT_DIR=/share/neurocomputation/Tim/git/2layer
mkdir -p "$PROJECT_DIR/1layer_version/out"

# ---- JAX on GPU: use node scratch for cache (avoids /tmp full) ----
if [ -n "$SLURM_TMPDIR" ]; then
  export TMPDIR="$SLURM_TMPDIR"
  echo "TMPDIR=$TMPDIR"
fi

# ---- XLA tuning for GPU (optional) ----
export XLA_FLAGS=--xla_gpu_autotune_level=4

# ---- Load conda ----
spack load miniconda3
source activate heid

# ---- SHD data path ----
export SHD_DATA_PATH="${SHD_DATA_PATH:-/share/neurocomputation/Tim/SHD_data}"

# ---- 1-layer SHD: overridable via env (edit here or export before sbatch) ----
export T_MS="${T_MS:-700}"
export N_HIDDEN="${N_HIDDEN:-64}"
export N_OUTPUTS="${N_OUTPUTS:-20}"
export RANDOM_SEED="${RANDOM_SEED:-12}"
export EPOCHS="${EPOCHS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export WARMUP_READOUT_EPOCHS="${WARMUP_READOUT_EPOCHS:-1}"
export LR_HIDDEN_DEND="${LR_HIDDEN_DEND:-0.045}"
export LR_HIDDEN_SOMA="${LR_HIDDEN_SOMA:-0.00015}"
export LR_READOUT="${LR_READOUT:-0.035}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
export GRADIENT_CLIP="${GRADIENT_CLIP:-5.0}"
export SPIKE_DROPOUT="${SPIKE_DROPOUT:-0.1}"
export LOSS_TEMPERATURE="${LOSS_TEMPERATURE:-2.7}"
export LOSS_COUNT_BIAS="${LOSS_COUNT_BIAS:-0.18}"
export LOSS_LABEL_SMOOTHING="${LOSS_LABEL_SMOOTHING:-0.13}"
# Input: set NO_KERNEL=1 for no alpha kernel; SPIKE_AMPLITUDE e.g. 1 or 5
export NO_KERNEL="${NO_KERNEL:-1}"
export SPIKE_AMPLITUDE="${SPIKE_AMPLITUDE:-1}"

echo "SHD_DATA_PATH=$SHD_DATA_PATH  RANDOM_SEED=$RANDOM_SEED  EPOCHS=$EPOCHS  BATCH_SIZE=$BATCH_SIZE"
echo "NO_KERNEL=$NO_KERNEL  SPIKE_AMPLITUDE=$SPIKE_AMPLITUDE"

# ---- Sanity check GPU & JAX ----
nvidia-smi
python -c "
import jax
print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
"

# ---- Run training (CLI args override env defaults above) ----
cd "$PROJECT_DIR"
EXTRA_ARGS=""
[ "$NO_KERNEL" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --no_kernel"
[ -n "$SPIKE_AMPLITUDE" ] && EXTRA_ARGS="$EXTRA_ARGS --spike_amplitude $SPIKE_AMPLITUDE"
python 1layer_version/run_shd.py \
  --T "$T_MS" \
  --n_hidden "$N_HIDDEN" \
  --n_outputs "$N_OUTPUTS" \
  --seed "$RANDOM_SEED" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --warmup_readout_epochs "$WARMUP_READOUT_EPOCHS" \
  --lr_hidden_dend "$LR_HIDDEN_DEND" \
  --lr_hidden_soma "$LR_HIDDEN_SOMA" \
  --lr_readout "$LR_READOUT" \
  --weight_decay "$WEIGHT_DECAY" \
  --gradient_clip "$GRADIENT_CLIP" \
  --spike_dropout "$SPIKE_DROPOUT" \
  --loss_temperature "$LOSS_TEMPERATURE" \
  --loss_count_bias "$LOSS_COUNT_BIAS" \
  --loss_label_smoothing "$LOSS_LABEL_SMOOTHING" \
  $EXTRA_ARGS

echo "Job finished at: $(date)"
