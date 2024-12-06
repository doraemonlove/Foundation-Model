# 添加开始时间记录
start_time=$(date +%s)

MODEL='logs/time_moe'
DATA='dataset/electricity/electricity.csv'

for PREDICTION_LENGTH in 1 24 48 96 256 352 512; 
do
python -u run_eval.py \
    --model $MODEL \
    --data $DATA \
    --batch_size 32 \
    --prediction_length $PREDICTION_LENGTH
done

# 添加结束时间计算和输出
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "总运行时间: $duration 秒"
echo "运行时间: $((duration/60))分 $((duration%60))秒"
