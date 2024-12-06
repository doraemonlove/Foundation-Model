#!/bin/bash

RUN_NAME="ecl_finetune"
MODEL="moirai_1.0_R_large"
DATA="ecl"
VAL_DATA="ecl"

python -m cli.train \
  -cp conf/finetune \
  run_name=$RUN_NAME \
  model=$MODEL \
  data=$DATA \
  val_data=$VAL_DATA