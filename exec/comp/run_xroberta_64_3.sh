#!/usr/bin/env bash
#SBATCH --job-name=xlm-roberta_64_3
#SBATCH -e output/slurm_%A.err
#SBATCH -o output/slurm_%A.out

#SBATCH --time=0-36:00:00
#SBATCH --account=desinformation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module purge
module load miniconda
module load gcc/11.3.0
conda activate rest_am


set -ex

# This scripts runs ACTA train and evaluation in the same process, but only uses
# a single device to avoid inconsistencies when evaluating

INPUT_DIR=./data/component_data/
# OUTPUT_DIR=./output/bert_e$e-b=$b-sl$MAX_SEQ_LENGTH-lr$LEARNING_RATE
CHECKPOINT_PATH=checkpoints
TASK_TYPE=seq-tag
MODEL=xlm-roberta
CACHE_DIR=./cache
EVALUATION_SPLIT=test
MAX_SEQ_LENGTH=64
LEARNING_RATE=3e-5
LABELS="O B-Claim I-Claim B-Premise I-Premise"
NUM_DEVICES=1
NUM_WORKERS=-1
LOG_STEPS=1000
SAVE_STEPS=2000
RANDOM_SEED=42

epochs=(1 2 3)
batch=(8 16 32)

for e in "${epochs[@]}"; do
  for b in "${batch[@]}"; do
    echo "Running with epochs=$e, batch=$b, max=$MAX_SEQ_LENGTH, lr=$LEARNING_RATE"
    echo "epoch=$e-batch=$b-seq-len=$MAX_SEQ_LENGTH-lr=$LEARNING_RATE"

    python ./run_acta.py \
      --input-dir $INPUT_DIR \
      --output-dir "./output/$MODEL-epoch=$e-batch=$b-seq-len=$MAX_SEQ_LENGTH-lr=$LEARNING_RATE" \
      --task-type $TASK_TYPE \
      --model $MODEL \
      --cache-dir $CACHE_DIR \
      --checkpoint-path $CHECKPOINT_PATH \
      --train \
      --evaluation-split $EVALUATION_SPLIT \
      --validation \
      --num-devices $NUM_DEVICES \
      --num-workers $NUM_WORKERS \
      --epochs $e \
      --train-batch-size $b \
      --eval-batch-size $b \
      --max-seq-length $MAX_SEQ_LENGTH \
      --learning-rate $LEARNING_RATE \
      --labels $LABELS \
      --lower-case \
      --log-every-n-steps $LOG_STEPS \
      --save-every-n-steps $SAVE_STEPS \
      --eval-all-checkpoints \
      --overwrite-output \
      --random-seed $RANDOM_SEED
  done
done


