#!/bin/bash

RUN_NAME="heating_eval_1"
MODEL="moirai_moe_1.0_R_base"
PATCH_SIZE=16
CONTEXT_LENGTH=512
for i in 12; do
DATA="heating_test_$i"

# 执行 Python 命令并传入参数
python -m cli.eval \
  run_name=$RUN_NAME \
  model=$MODEL \
  model.patch_size=$PATCH_SIZE \
  model.context_length=$CONTEXT_LENGTH \
  data=$DATA
done