#!/bin/bash

RUN_NAME="ecl_finetune"
MODEL="moirai_1.0_R_large"
DATA="ecl"
VAL_DATA="ecl"

# 添加debugpy配置
python -m cli.train \
  -cp conf/finetune \
  run_name=$RUN_NAME \
  model=$MODEL \
  data=$DATA \
  val_data=$VAL_DATA
  

# RUN_NAME="ecl_eval_2"
# MODEL="moirai_1.0_R_large_finetune"
# PATCH_SIZE=16
# CONTEXT_LENGTH=512
# for i in 96 256 512; do
# DATA="ecl_test_$i"

# # 执行 Python 命令并传入参数
# python -m cli.eval \
#   run_name=$RUN_NAME \
#   model=$MODEL \
#   model.patch_size=$PATCH_SIZE \
#   model.context_length=$CONTEXT_LENGTH \
#   data=$DATA
# done
