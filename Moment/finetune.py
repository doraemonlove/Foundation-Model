import os
import json  # 导入 JSON 模块
# 设置 Hugging Face 镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from momentfm import MOMENTPipeline
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from momentfm.utils.utils import control_randomness
import argparse

# 常量定义
MAX_LR = 1e-4
MAX_NORM = 5.0


# 数据处理部分
class CollectData:
    def __init__(self, forecast_horizon, data_path, data_split, data_stride_len=1, task_name="forecasting", random_seed=42, train_pct=0.7, val_pct=0.15, test_pct=0.15, context_length=512):
        self.seq_len = context_length
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = data_path
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self._read_data()

    def _get_borders(self, train_pct, val_pct, test_pct):
        total_length = self.length_timeseries_original
        n_train = int(total_length * train_pct)
        n_val = int(total_length * val_pct)
        n_test = int(total_length * test_pct)
        assert n_train + n_val + n_test <= total_length, "Total split exceeds the length of the dataset."
        return slice(0, n_train), slice(n_train + n_val, total_length)

    def _read_data(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(self.full_file_path_and_name)
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        df.drop(columns=["date"], inplace=True)
        df = df.infer_objects(copy=False).interpolate(method="cubic")
        data_splits = self._get_borders(self.train_pct, self.val_pct, self.test_pct)

        train_data = df[data_splits[0]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        self.data = df[data_splits[0], :] if self.data_split == "train" else df[data_splits[1], :]
        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if self.task_name == "forecasting":
            pred_end = seq_end + self.forecast_horizon
            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T
            forecast = self.data[seq_end:pred_end, :].T
            return timeseries, forecast, input_mask

    def __len__(self):
        return (self.length_timeseries - self.seq_len - self.forecast_horizon) // self.data_stride_len + 1

# 评估指标
def mse(trues, preds): return np.mean((trues - preds) ** 2)
def mae(trues, preds): return np.mean(np.abs(trues - preds))
def smape(trues, preds):
    numerator = np.abs(trues - preds)
    denominator = (np.abs(trues) + np.abs(preds)) / 2
    return 200 * np.mean(numerator / denominator)

def mape(trues, preds): return np.mean(np.abs((trues - preds) / trues)) * 100
def mase(trues, preds, historical_values):
    naive_error = np.abs(trues[1:] - trues[:-1])
    forecast_error = np.abs(trues - preds)
    return np.mean(forecast_error[1:]) / np.mean(naive_error)

# 主函数
def main():
    torch.cuda.empty_cache()
    control_randomness(seed=42)

    parser = argparse.ArgumentParser(description='Fine-tune MOMENT model.')
    parser.add_argument('--forecast_horizon', type=int, default=512, help='Forecast horizon')
    parser.add_argument('--data_path', type=str, default='./data/electricity.csv', help='Path to the data file')
    parser.add_argument('--model_path', type=str, default='AutonLab/MOMENT-1-large', help='Path to the pre-trained model')
    parser.add_argument('--output_path', type=str, default='./output', help='Path to save the output model')
    parser.add_argument('--max_epoch', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training')
    parser.add_argument('--split_config', type=str, default='{"train": 0.7, "val": 0.15, "test": 0.15}', help='JSON string for train/val/test split ratios')
    parser.add_argument('--context_length', type=int, default=512, help='Context length for the model')
    parser.add_argument('--fewshot_ratio', type=float, default=1.0, help='Ratio of training data to use for Fewshot learning')
    args = parser.parse_args()

    # 解析 split_config 参数
    split_config = json.loads(args.split_config)
    train_pct = split_config.get("train", 0.7)
    val_pct = split_config.get("val", 0.15)
    test_pct = split_config.get("test", 0.15)

    print("开始数据集准备...")  # 提示信息
    train_dataset = CollectData(
        forecast_horizon=args.forecast_horizon,
        data_path=args.data_path,
        data_split="train",
        random_seed=13,
        train_pct=train_pct * args.fewshot_ratio,
        val_pct=val_pct,
        test_pct=test_pct,
        context_length=args.context_length
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = CollectData(forecast_horizon=args.forecast_horizon,
                            data_path=args.data_path, 
                            data_split="test", 
                            random_seed=13,
                            train_pct=train_pct * args.fewshot_ratio,
                            val_pct=val_pct,
                            test_pct=test_pct,
                            context_length=args.context_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    print("数据集准备完成。")  # 提示信息
    print("-" * 50)  # 分隔符

    print("开始模型初始化...")  # 提示信息
    model = MOMENTPipeline.from_pretrained(
        args.model_path,
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': args.forecast_horizon,
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': True,
            'freeze_embedder': True,
            'freeze_head': False,
        },
        local_files_only=True
    )
    model.init()
    print("模型初始化完成。")  # 提示信息
    print("-" * 50)  # 分隔符

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 不再选择特定 GPU
    model = model.to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=len(train_loader) * args.max_epoch, pct_start=0.3)

    print("开始训练模型...")  # 提示信息
    for epoch in range(args.max_epoch):
        model.train()
        losses = []
        for timeseries, forecast, input_mask in tqdm(train_loader, total=len(train_loader)):
            timeseries = timeseries.float().to(device).requires_grad_()
            forecast = forecast.float().to(device)
            input_mask = input_mask.to(device)

            with torch.cuda.amp.autocast():
                output = model(x_enc=timeseries, input_mask=input_mask)
            
            loss = criterion(output.forecast, forecast)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

        scheduler.step()
        print(f"Epoch {epoch}: Train loss: {np.mean(losses):.6f}")

    print("训练完成。")  # 提示信息
    print("-" * 50)  # 分隔符

    # 保存模型权重
    torch.save(model.state_dict(), args.output_path)
    print(f"模型权重已保存到: {args.output_path}")

    print("开始测试模型...")  # 提示信息
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for timeseries, forecast, input_mask in tqdm(test_loader, total=len(test_loader)):
            timeseries = timeseries.float().to(device).requires_grad_()
            forecast = forecast.float().to(device)
            input_mask = input_mask.to(device)

            # with torch.amp.autocast('cuda'):
            #     output = torch.utils.checkpoint.checkpoint(model, timeseries, input_mask, use_reentrant=False)
            with torch.cuda.amp.autocast():
                output = model(x_enc=timeseries, input_mask=input_mask)

            trues.append(forecast.cpu().numpy())
            preds.append(output.forecast.cpu().numpy())

    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)

    print("测试完成。")  # 提示信息
    print("-" * 50)  # 分隔符
    print(f"MSE: {mse(trues, preds):.6f}")
    print(f"MAE: {mae(trues, preds):.6f}")
    print(f"SMAPE: {smape(trues, preds):.6f}")
    print(f"MAPE: {mape(trues, preds):.6f}")
    print(f"MASE: {mase(trues, preds, trues[:-1]):.6f}")

if __name__ == "__main__":
    main()

"""
运行命令示例：
CUDA_VISIBLE_DEVICES=2 python finetune.py --forecast_horizon 96 \
      --data_path './data/electricity.csv' \
      --model_path '/home/zhupengtian/zhangqingliang/models/MOMENT-1-large' \
      --output_path './output' \
      --max_epoch 3 \
      --batch_size 8 \
      --split_config '{"train": 0.7, "val": 0.15, "test": 0.15}' \
      --context_length 512 \
      --fewshot_ratio 0.1
"""