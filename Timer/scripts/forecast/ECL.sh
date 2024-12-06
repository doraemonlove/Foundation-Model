#!/bin/sh
# 添加开始时间记录
start_time=$(date +%s)

# zero-shot
# model_name=Timer
# seq_len=480
# pred_len=16
# label_len=$((seq_len - pred_len))
# ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
# data=electricity
# subset_rand_ratio=1

# torchrun --nnodes=1 --nproc_per_node=1 run.py \
#   --task_name forecast \
#   --is_training 0 \
#   --ckpt_path $ckpt_path \
#   --root_path ./dataset/$data/ \
#   --data_path $data.csv \
#   --data custom \
#   --model_id electricity_sr_$subset_rand_ratio \
#   --model $model_name \
#   --features M \
#   --seq_len $seq_len \
#   --label_len $label_len \
#   --pred_len $pred_len \
#   --output_len_list 16 48 96 128 256 352 480 \
#   --patch_len 96 \
#   --e_layers 8 \
#   --factor 3 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --d_ff 2048 \
#   --batch_size 256 \
#   --learning_rate 3e-5 \
#   --num_workers 1 \
#   --train_test 0 \
#   --subset_rand_ratio $subset_rand_ratio \
#   --itr 1 \
#   --gpu 0 \
#   --use_ims \
#   --is_finetuning 0 


# few-shot test
# model_name=Timer
# seq_len=480
# pred_len=16
# label_len=$((seq_len - pred_len))
# data=electricity
# subset_rand_ratio=1

# torchrun --nnodes=1 --nproc_per_node=1 run.py \
#   --task_name forecast \
#   --is_training 0 \
#   --root_path ./dataset/$data/ \
#   --data_path $data.csv \
#   --data custom \
#   --model_id electricity_sr_$subset_rand_ratio \
#   --model $model_name \
#   --features M \
#   --seq_len $seq_len \
#   --label_len $label_len \
#   --pred_len $pred_len \
#   --output_len_list 16 48 96 128 256 352 480 \
#   --patch_len 96 \
#   --e_layers 8 \
#   --factor 3 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --d_ff 2048 \
#   --batch_size 256 \
#   --learning_rate 3e-5 \
#   --num_workers 1 \
#   --train_test 0 \
#   --subset_rand_ratio $subset_rand_ratio \
#   --itr 1 \
#   --gpu 0 \
#   --use_ims \
#   --is_finetuning 0 \
#   --use_finetune_model 1 \
#   --checkpoints './checkpoints/forecast_electricity_sr_1_Timer_custom_ftM_sl480_ll384_pl96_pl96_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp/checkpoint.pth'

# few-shot train
model_name=Timer
seq_len=480
pred_len=96
label_len=$((seq_len - pred_len))
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=electricity
subset_rand_ratio=1

torchrun --nnodes=1 --nproc_per_node=1 run.py \
  --task_name forecast \
  --is_training 0 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/$data/ \
  --data_path $data.csv \
  --data custom \
  --model_id electricity_sr_$subset_rand_ratio \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --patch_len 96 \
  --e_layers 8 \
  --factor 3 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 256 \
  --learning_rate 3e-5 \
  --num_workers 1 \
  --train_test 0 \
  --subset_rand_ratio $subset_rand_ratio \
  --itr 1 \
  --gpu 0 \
  --use_ims \
  --is_finetuning 1

# 添加结束时间计算和输出
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "总运行时间: $duration 秒"
echo "运行时间: $((duration/60))分 $((duration%60))秒"