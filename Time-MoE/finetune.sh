# 添加开始时间记录
start_time=$(date +%s)

MODEL='./TimeMoE-200M'
DATA='dataset/electricity'

python -u main.py \
    --model_path $MODEL \
    --data_path $DATA \
    --num_train_epochs 5 \
    --global_batch_size 32

# 添加结束时间计算和输出
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "总运行时间: $duration 秒"
echo "运行时间: $((duration/60))分 $((duration%60))秒"