export CUDA_VISIBLE_DEVICES=1

d_model=128
#ckpt_path=checkpoints/ALL_task_fewshot_newdata_finetune_pct20_UniTS_All_ftM_dm128_el3_Exp_0/checkpoint.pth
ckpt_path=checkpoints/ALL_task_UniTS_zeroshot_pretrain_x128_UniTS_zeroshot_All_ftM_dm128_el3_Exp_0/pretrain_checkpoint.pth
python evaluate.py \
  --is_finetuning 0 \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model $d_model \
  --des 'Exp' \
  --pretrained_weight $ckpt_path \
  --task_data_config_path  data_provider/zeroshot_task.yaml \
  --output_dir ./results/zeroshot/multivariate_13