import torch
import pandas as pd
import argparse
import taos
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
import numpy as np
import datetime
import json
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv

taos_host = ''
taos_user = ''
taos_password = ''
taos_database = ''
PRED_WINDOW = datetime.timedelta(hours=4)
INTERVAL = datetime.timedelta(minutes=3)
PRED_LEN = PRED_WINDOW // INTERVAL
ENCODING = "UTF-8"

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  

metrics = [
    'rpst', 'rprt', 'rpsif', 'rpsih',
    'rsst', 'rsrt', 'rssif', 'rssih',
    'acpst', 'acprt', 'acpsif', 'acpsih',
    'acsst', 'acsrt',
    'ot', 'oh', 'oli'
]

class TaosAPI:
    def __init__(self, host='localhost', user='root', password='taosdata', database='test'):
        # 初始化连接信息
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.conn = taos.connect(host=self.host, user=self.user, password=self.password, database=self.database)
        self.cursor = self.conn.cursor()

class MoiraiMoE:
    def __init__(self, args):
        self.args = args
        self.model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(self.args.model),
            prediction_length=self.args.prediction_length,
            context_length=self.args.context_length,
            patch_size=self.args.patch_size,
            num_samples=self.args.num_samples,
            target_dim=17,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0
        )

    def predict(self, input_data):
        forecasts = self.model(
            past_target=input_data['past_target'],
            past_observed_target=input_data['past_observed_target'],
            past_is_pad=input_data['past_is_pad']
        )
        return forecasts

def load_datafile(
    filename: str,
    start: datetime.datetime,
    end: datetime.datetime,
    interval: datetime.timedelta = datetime.timedelta(minutes=3),
):
    raw_data: pd.DataFrame = pd.read_csv(
        filename,
        parse_dates=True,
        index_col="ts",
    )
    raw_data = raw_data[metrics]

    data = raw_data.resample(interval, origin="start").mean()
    
    return (
        data.iloc[(data.index >= start) & (data.index <= end)],
        data.iloc[data.index < start],
    )

def load_param_data(
    data_dir: str,
    label_filename: str,
    window: datetime.timedelta = PRED_WINDOW,
    interval: datetime.timedelta = INTERVAL,
):
    with open(label_filename, encoding=ENCODING) as obj:
        labels = json.load(obj)

    data = []
    for label in labels:
        changed_time = datetime.datetime.fromisoformat(label["time"])
        name = label['ticket_id'].split('_')[0]
        within, before = load_datafile(
            filename=os.path.join(data_dir, f"{name}"),
            start=changed_time,
            end=changed_time + window,
            interval=interval,
        )
        if within.shape[0] >= PRED_LEN and before.shape[0] >= PRED_LEN:
            # 转换数据格式为模型所需的输入格式
            before_values = before.iloc[-PRED_LEN:].values
            before_values = torch.as_tensor(before_values, dtype=torch.float32)
            before_values = before_values.reshape(1, -1, before_values.shape[1])
            
            # 创建观察掩码和填充掩码
            past_observed_target = torch.ones_like(before_values, dtype=torch.bool)
            past_is_pad = torch.zeros((1, before_values.shape[1]), dtype=torch.bool)
            
            past_dict = {
                "past_target": before_values,
                "past_observed_target": past_observed_target,
                "past_is_pad": past_is_pad
            }
            
            data.append({
                "past": past_dict,
                "before": before.iloc[-PRED_LEN:],
                "within": within.iloc[:PRED_LEN]
            })
    
    print(f'>>> Loaded {len(data)} data')
    return data

def collect_data(start_time=None, end_time=None, context_length=None):
    data_path = 'dataset/electricity/electricity.csv'
    target = 'OT'
    df = pd.read_csv(data_path)
    
    # 转换数据格式为模型所需的输入格式
    target_values = df[[target]].values[:context_length]
    target_values = torch.as_tensor(target_values, dtype=torch.float32)
    target_values = target_values.reshape(1, -1, 1)  # 调整维度为 (batch, time, variate)
    
    # 创建观察掩码和填充掩码
    past_observed_target = torch.ones_like(target_values, dtype=torch.bool)
    past_is_pad = torch.zeros_like(target_values, dtype=torch.bool).squeeze(-1)
    
    return {
        "past_target": target_values,
        "past_observed_target": past_observed_target,
        "past_is_pad": past_is_pad
    }

# 添加预测结果处理方法字典
FORECAST_METHODS = {
    'median': lambda x: np.round(np.median(x, axis=0), decimals=4),
    'mean': lambda x: np.round(np.mean(x, axis=0), decimals=4)
}

def sample_data(preds, method='median'):
    """
    处理并保存预测结果
    
    Args:
        preds: 预测结果，形状为 (1, num_samples, seq_len, num_metrics)
        method: 结果处理方法，可选 'median', 'mean'
        output_path: 输出路径
    """
    if method not in FORECAST_METHODS:
        raise ValueError(f"不支持的方法: {method}. 可用方法: {list(FORECAST_METHODS.keys())}")
    
    # 使用指定方法处理样本维度
    processed_preds = FORECAST_METHODS[method](preds[0])
    
    # 创建DataFrame
    df = pd.DataFrame(processed_preds, columns=metrics)
    
    # 定义泵频率相关的列
    pump_freq_cols = ['rpf1', 'rpf2', 'rpf3', 'acpf1', 'acpf2', 'acpf3', 'acpf4']
    
    # 处理每一列的负值
    for col in df.columns:
        negative_mask = df[col] < 0
        if negative_mask.any():
            print(f"列 {col} 发现 {negative_mask.sum()} 个负值")
            
            if col in pump_freq_cols:
                # 对于泵频率指标，直接将负值替换为0
                df.loc[negative_mask, col] = 0
            else:
                # 其他指标使用插值逻辑
                df.loc[negative_mask, col] = np.nan
                df[col] = df[col].interpolate(method='linear')
                df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
    
    return df

def plot_metrics_to_pdf(before_df: pd.DataFrame, pred_df: pd.DataFrame, gt_df: pd.DataFrame, pdf_path: str):
    """为每个指标创建历史数据和预测对比图，同一张图中前半段显示历史数据，后半段显示预测值和真实值"""
    
    with PdfPages(pdf_path) as pdf:
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(15, 6))
            
            # 获取时间点数量
            before_len = len(before_df)
            pred_len = len(pred_df)
            
            # 绘制历史数据（前半部分）
            ax.plot(range(before_len), before_df[metric], 
                   label='历史数据', color='gray')
            
            # 绘制预测值和真实值（后半部分）
            x_pred = range(before_len, before_len + pred_len)
            ax.plot(x_pred, pred_df[metric], 
                   label='预测值', color='red')
            ax.plot(x_pred, gt_df[metric], 
                   label='真实值', color='blue')
            
            # 添加垂直分界线
            ax.axvline(x=before_len, color='black', linestyle='--', alpha=0.5)
            
            ax.set_title(f'指标: {metric}')
            ax.set_xlabel('时间步')
            ax.set_ylabel('数值')
            ax.legend()
            ax.grid(True)
            
            # 保存到PDF
            pdf.savefig(fig)
            plt.close(fig)

def _maximum_error(preds: np.ndarray, target: np.ndarray) -> float:
    return abs(np.nanmax(preds) - np.nanmax(target))


def _mean_error(preds: np.ndarray, target: np.ndarray) -> float:
    return abs(np.nanmean(preds) - np.nanmean(target))


def _rooted_mean_squared_error(preds: np.ndarray, target: np.ndarray) -> float:
    return np.sqrt(np.nanmean((preds - target) ** 2))


def _symmetric_mean_absolute_percentage_error(
    preds: np.ndarray, target: np.ndarray
) -> float:
    return np.nanmean(abs((preds - target) / ((preds + target) / 2)))


_EVALUATION_METRICS = (
    ("MaxE", _maximum_error),
    ("MeanE", _mean_error),
    ("RMSE", _rooted_mean_squared_error),
    ("sMAPE", _symmetric_mean_absolute_percentage_error),
)

def save_evaluation_results(results_dict, output_dir):
    """保存评估结果到CSV文件
    
    Args:
        results_dict: 包含评估结果的字典，格式为：
            {sample_index: {metric: {evaluation_metric: score}}}
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'evaluation_results.csv')
    
    with open(output_path, 'w', encoding=ENCODING) as f:
        writer = csv.writer(f)
        
        # 写入表头
        headers = ['样本序号']
        for metric in metrics:
            for eval_metric, _ in _EVALUATION_METRICS:
                headers.append(f'{metric}_{eval_metric}')
        writer.writerow(headers)
        
        # 写入数据
        for sample_idx, sample_results in results_dict.items():
            row = [sample_idx]
            for metric in metrics:
                for eval_metric, _ in _EVALUATION_METRICS:
                    row.append(sample_results[metric][eval_metric])
            writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Moirai Predict')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='model/moirai-moe-1.0-R-base',
        help='Model path'
    )
    parser.add_argument(
        '--start_time',
        type=int,
        default=None,
        help='Start time'
    )

    parser.add_argument(
        '--end_time',
        type=int,
        default=None,
        help='End time'
    )

    parser.add_argument(
        '--context_length',
        type=int,
        default=32,
        help='Context length'
    )

    parser.add_argument(
        '--prediction_length', '-p',
        type=int,
        default=16,
        help='Prediction length'
    )

    parser.add_argument(
        '--patch_size',
        type=int,
        default=16,
        help='Patch size'
    )

    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of samples'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output dir'
    )

    parser.add_argument(
        '--forecast_method',
        type=str,
        default='median',
        choices=list(FORECAST_METHODS.keys()),
        help='预测结果处理方法'
    )

    args = parser.parse_args()

    device = 'cpu'

    # input_data = collect_data(args.start_time, args.end_time, args.context_length)
    input_data = load_param_data('dataset/heating', 'dataset/heating/labels-1230-test.json')
    
    model = MoiraiMoE(args)

    evaluation_results = {}
    for index, data in enumerate(input_data):
        preds = model.predict(data['past'])
        output_path = os.path.join(args.output_dir, f'{index}.pdf')
        pred_df = sample_data(preds, method=args.forecast_method)
        plot_metrics_to_pdf(data['before'], pred_df, data['within'], output_path)
        
        # 收集评估结果
        sample_results = {}
        for metric in metrics:
            pred_values = pred_df[metric].values
            true_values = data['within'][metric].values
            
            metric_results = {}
            for metric_name, metric_fn in _EVALUATION_METRICS:
                score = metric_fn(pred_values, true_values)
                metric_results[metric_name] = score
            sample_results[metric] = metric_results
        
        evaluation_results[index] = sample_results
    
    # 保存评估结果
    save_evaluation_results(evaluation_results, args.output_dir)