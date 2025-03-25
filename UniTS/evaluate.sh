export CUDA_VISIBLE_DEVICES=1
model_name=UniTS_fewshot_newdata
exp_name=UniTS_fewshot_newdata
wandb_mode=disabled
ptune_name=fewshot_newdata

d_model=128
ckpt_path=checkpoints/ALL_task_fewshot_newdata_finetune_pct20_UniTS_All_ftM_dm128_el3_Exp_0/checkpoint.pth
python evaluate.py \
  --is_training 0 \
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model $d_model \
  --des 'Exp' \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --pretrained_weight $ckpt_path \
  --task_data_config_path  data_provider/fewshot_new_task.yaml \
  --output_dir ./results/finetune