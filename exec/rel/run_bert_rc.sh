#!/usr/bin/env bash
#SBATCH --job-name=bert-rc
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

INPUT_DIR=./data/relation_data/
# OUTPUT_DIR=./output/bert_e$e-b=$b-sl$MAX_SEQ_LENGTH-lr$LEARNING_RATE
CHECKPOINT_PATH=checkpoints
TASK_TYPE=rel-class
MODEL=bert
CACHE_DIR=./cache
EVALUATION_SPLIT=test
MAX_SEQ_LENGTH=32
LEARNING_RATE=1e-5
# LABELS="O B-Claim I-Claim B-Premise I-Premise"
NUM_DEVICES=1
NUM_WORKERS=-1
LOG_STEPS=250
SAVE_STEPS=500
RANDOM_SEED=42

epochs=(3)
batch=(16)
max=(256)
lr=(4e-05)

for e in "${epochs[@]}"; do
  for b in "${batch[@]}"; do
    for m in "${max[@]}"; do
      for l in "${lr[@]}"; do
        echo "Running with epochs=$e, batch=$b, max=$m, lr=$l"
        echo "epoch=$e-batch=$b-seq-len=$m-lr=$l"

        python ./run_acta.py \
          --input-dir $INPUT_DIR \
          --output-dir "./output/$MODEL-epoch=$e-batch=$b-$TASK_TYPE=$m-lr=$l" \
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
          --max-seq-length $m \
          --learning-rate $l \
          --lower-case \
          --log-every-n-steps $LOG_STEPS \
          --save-every-n-steps $SAVE_STEPS \
          --eval-all-checkpoints \
          --overwrite-output \
          --random-seed $RANDOM_SEED \
          --weighted-loss
      done
    done
  done
done