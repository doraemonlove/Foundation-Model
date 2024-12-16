import argparse
import pandas as pd
import torch
import numpy as np
import taos  
from tqdm import tqdm
from src.chronos.chronos_bolt import ChronosBoltPipeline, ChronosBoltConfig
import os

# 设置 Hugging Face镜像网址环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

taos_host = ''
taos_user = ''
taos_password = ''
taos_database = ''

class TaosAPI:
    def __init__(self, host='localhost', user='root', password='taosdata', database='test'):
        # 初始化连接信息
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.conn = taos.connect(host=self.host, user=self.user, password=self.password, database=self.database)
        self.cursor = self.conn.cursor()


class ChronosPredictor:
    def __init__(self, args):
        self.args = args
        print("Loading model...")
        with tqdm(total=1, desc="Loading model") as pbar:
            self.model = ChronosBoltPipeline.from_pretrained(self.args.model)
            self.model.model.to(self.args.device)
            self.model.model.eval()

            pbar.update(1)

    def predict(self, input_data):
        """
        执行预测

        Args:
            input_data: 包含 past_target 的字典

        Returns:
            预测结果样本
        """
        print("\nPerforming prediction...")
        with torch.no_grad(), tqdm(total=1, desc="Predicting") as pbar:
            # 使用 predict 方法获取样本预测
            samples = self.model.predict(
                context=input_data['past_target'],
                prediction_length=self.args.prediction_length,
                # quantile_levels=[0.1, 0.5, 0.9],# 添加
            )
            pbar.update(1)

        return {
            'samples': samples
        }

def collect_data(args):
    """
    收集并准备输入数据
    """
    print("\nCollecting data...")
    with tqdm(total=3, desc="Loading data") as pbar:
        data_path = args.data_path
        df = pd.read_csv(data_path)

        start_time = args.start_time
        end_time = args.end_time
        context_length = args.context_length

        # 如果指定了时间范围，进行过滤
        if start_time is not None and end_time is not None:
            df = df[(df.index >= start_time) & (df.index <= end_time)]

        pbar.update(1)

        # 如果没有指定 context_length，使用所有可用数据
        if context_length is None:
            context_length = len(df)

        if not args.target_column:
            # 假设时间列的名称为 'timestamp' 或者 'date'
            time_columns = ['timestamp', 'date']
            non_time_columns = [col for col in df.columns if col not in time_columns]

            # 提取所有非时间列的值并拼接成一个张量
            all_values = []
            for column in non_time_columns:
                column_values = df[column].values[-context_length:]  # 最近的 context_length 数据
                all_values.append(column_values)

            # 将所有序列沿列方向拼接为一个 2D 张量
            target_values = torch.as_tensor(all_values, dtype=torch.float32).T
        else:
            # 提取指定的目标列
            target = args.target_column
            target_values = df[[target]].values[-context_length:]  # 最近的 context_length 数据
            target_values = torch.as_tensor(target_values, dtype=torch.float32)

        # 确保返回的张量为 2D
        target_values = target_values.unsqueeze(0) if target_values.ndim == 1 else target_values
        pbar.update(2)

        return {
            "past_target": target_values,
        }



# 修改预测结果处理方法字典
FORECAST_METHODS = {
    'raw_samples': lambda x: np.round(x['samples'].numpy(), decimals=4),
    'mean': lambda x: np.round(x['samples'].mean(dim=1).numpy(), decimals=4),
    'median': lambda x: np.round(np.median(x['samples'].numpy(), axis=1), decimals=4),
    'all_stats': lambda x: {
        'samples': np.round(x['samples'].numpy(), decimals=4),
        'mean': np.round(x['samples'].mean(dim=1).numpy(), decimals=4),
        'median': np.round(np.median(x['samples'].numpy(), axis=1), decimals=4)
    }
}

def save_data(preds, method='median', output_path=None):
    """
    处理并保存预测结果
    """
    print("\nProcessing and saving predictions...")
    with tqdm(total=2, desc="Saving results") as pbar:
        if method not in FORECAST_METHODS:
            raise ValueError(f"不支持的方法: {method}. 可用方法: {list(FORECAST_METHODS.keys())}")

        processed_preds = FORECAST_METHODS[method](preds)
        pbar.update(1)

        if output_path:
            if isinstance(processed_preds, dict):
                np.savez(output_path, **processed_preds)
            else:
                np.save(output_path, processed_preds)
        pbar.update(1)

        print(f'\n>>> Processed Preds ({method}): ', processed_preds)

    return processed_preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ChronosBolt Predict')

    # 处理输入数据
    parser.add_argument('--start_time', type=int, default=None, help='Start time')
    parser.add_argument('--end_time', type=int, default=None, help='End time')
    parser.add_argument('--context_length', type=int, default=512, help='Context length')
    parser.add_argument('--prediction_length', '-p', type=int, default=64, help='Prediction length')
    parser.add_argument('--data_path', type=str, default='dataset/electricity/electricity.csv', help='Path to the input data file')
    parser.add_argument('--target_column', type=str, default=None, help='Name of the target column in the data file')
    parser.add_argument('--time_column', type=str, default='date', help='Name of the timestamp column in the data file')

    # 模型本身参数
    parser.add_argument('--model', '-m', type=str, default='amazon/chronos-bolt-tiny', help='Model path or name')
    
    # 显卡设置
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')

    # 处理输出
    parser.add_argument('--forecast_method', type=str, default='raw_samples', choices=list(FORECAST_METHODS.keys()), help='预测结果处理方法: raw_samples(原始样本), mean(均值), median(中位数), all_stats(所有统计)')
    parser.add_argument('--output_path', type=str, default='./output', help='Path to the output data file')

    args = parser.parse_args()

    if args.device.isdigit():  # 如果传入的是数字
        args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    print("\nStarting prediction pipeline...")
    print(f"Using device: {args.device}")

    # 准备输入数据
    input_data = collect_data(args)

    # 初始化模型并预测
    model = ChronosPredictor(args)
    preds = model.predict(input_data)

    # 保存预测结果
    save_data(preds, method=args.forecast_method, output_path=args.output_path)

    print("\nPrediction completed successfully!")

    """
    示例：
python predict.py --device 3 \
    --data_path /home/zhupengtian/zhangqingliang/datasets/electricity.csv \
    --prediction_length 16
    """