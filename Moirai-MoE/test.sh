#!/bin/bash

RUN_NAME="heating_eval_zeroshot"
BASE_MODEL="moirai_moe_1.0_R_base"
PATCH_SIZE=16
CONTEXT_LENGTH=256
for i in 32 64 128 256; do
DATA="heating_test_$i"

python -m cli.eval \
  run_name=$RUN_NAME \
  model=$BASE_MODEL \
  model.patch_size=$PATCH_SIZE \
  model.context_length=$CONTEXT_LENGTH \
  data=$DATA
done