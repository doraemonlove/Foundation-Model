import torch
import pandas as pd
import argparse
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
import numpy as np
import datetime
import json
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
import traceback
ENCODING = 'utf-8'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  
metrics = [
    'rpst', 'rprt', 'rpsif', 'rpsih',
    'rsst', 'rsrt', 'rssif', 'rssih',
    'acpst', 'acprt', 'acpsif', 'acpsih',
    'msih',
    'rpvo1', 'rpvo2', 'acpvo1', 'acpvo2',
    'rpf1', 'rpf2', 'rpf3', 'acpf1', 'acpf2', 'acpf3', 'acpf4'
]

class MoiraiMoE:
    def __init__(self, args):
        self.args = args
        self.model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(self.args.model),
            prediction_length=self.args.prediction_length,
            context_length=self.args.context_length,
            patch_size=self.args.patch_size,
            num_samples=self.args.num_samples,
            target_dim=self.args.target_dim,
            feat_dynamic_real_dim=self.args.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=self.args.past_feat_dynamic_real_dim
        )

    def predict(self, input_data):
        self.model.to(self.args.device)
        self.model.eval()
        with torch.no_grad():
            forecasts = self.model(
                past_target=input_data['past_target'],
                past_observed_target=input_data['past_observed_target'],
                past_is_pad=input_data['past_is_pad']
            )
        return forecasts
    

def load_feather_data(
    feather_path: str,
    context_length: int,
    prediction_length: int,
    batch_size: int = 1000,  # 多少行数据作为一个 batch
    device: str = "cpu"  # 是否运行在 GPU
):
    """优化从 Feather 文件加载数据，将每行转换成 (context_length, n_features) 格式，并支持 GPU 计算"""
    
    # 读取 Feather 文件
    df = pd.read_feather(feather_path)
    print(f">>> 读取数据: {df.shape[0]} 个样本, {df.shape[1]} 个指标")

    # 如果存在 'rpsih' 和 'acpsih'，添加 'msih'
    if 'rpsih' in df.columns and 'acpsih' in df.columns:
        df['msih'] = df['rpsih'] + df['acpsih']
    # 只保留需要的 `metrics` 列
    df = df[metrics]

    data = []
    
    for start_idx in range(0, len(df), batch_size):
        batch_df = df.iloc[start_idx : start_idx + batch_size]

        # 将每个 `list` 变成 NumPy 数组，并转换形状
        batch_np = np.array(batch_df.applymap(lambda x: np.array(x, dtype=np.float32)).values.tolist())
        
        # `batch_np.shape` 原本是 `(batch_size, num_features, context_length + prediction_length)`
        batch_np = np.transpose(batch_np, (0, 2, 1))  # 变为 `(batch_size, context_length + prediction_length, num_features)`

        # 拆分 `before`（历史数据）和 `after`（真实值）
        before_values = batch_np[:, :context_length, :]  # `(batch_size, context_length, num_features)`
        after_values = batch_np[:, context_length: context_length + prediction_length, :]  # `(batch_size, prediction_length, num_features)`

        # 转换为 PyTorch Tensor，并移动到设备（CPU/GPU）
        before_values = torch.tensor(before_values, dtype=torch.float32, device=device)
        after_values = torch.tensor(after_values, dtype=torch.float32, device=device)

        past_dict = {
            "past_target": before_values,
            "past_observed_target": torch.ones_like(before_values, dtype=torch.bool, device=device),
            "past_is_pad": torch.zeros((before_values.shape[0], context_length), dtype=torch.bool, device=device),
        }

        # DataFrame 格式的 `before` 和 `after`
        before_df = batch_df.applymap(lambda x: x[:context_length])
        after_df = batch_df.applymap(lambda x: x[context_length: context_length + prediction_length])

        data.append({
            "past": past_dict,  # 供模型使用的历史数据
            "before": before_df,  # DataFrame 格式的历史数据
            "after": after_df,  # DataFrame 格式的真实值
            "ground_truth": after_values,  # PyTorch 格式的真实值
        })
    
    print(f'>>> Loaded {len(data)} batches of size {batch_size}')
    return data



def sample_data(preds, method='median'):
    """
    处理并保存预测结果
    
    Args:
        preds: 预测结果，形状为 (batch_size, num_samples, pred_len, num_metrics)
        method: 结果处理方法，可选 'median', 'mean'
    
    Returns:
        处理后的 DataFrame
    """
    if method not in FORECAST_METHODS:
        raise ValueError(f"不支持的方法: {method}. 可用方法: {list(FORECAST_METHODS.keys())}")

    # 计算 `num_samples` 维度的统计量
    processed_preds = FORECAST_METHODS[method](preds, axis=1)  # 在 num_samples 维度计算

    # 将 `batch_size` 和 `pred_len` 合并，适配 DataFrame
    batch_size, pred_len, num_metrics = processed_preds.shape[0], processed_preds.shape[1], processed_preds.shape[2]
    processed_preds = processed_preds.reshape(batch_size * pred_len, num_metrics)

    # 创建 DataFrame
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

# 定义新的 FORECAST_METHODS 适配 `batch_size` 维度
FORECAST_METHODS = {
    'median': lambda x, axis: np.round(np.median(x, axis=axis), decimals=4),
    'mean': lambda x, axis: np.round(np.mean(x, axis=axis), decimals=4),
}

def _mean_absolute_error(pred_values, true_values):
    return np.mean(np.abs(pred_values - true_values))


def _mean_squared_error(pred_values, true_values):
    return np.mean((pred_values - true_values) ** 2)


def _mean_absolute_percentage_error(pred_values, true_values):
    # 避免除以零
    mask = true_values != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((pred_values[mask] - true_values[mask]) / true_values[mask])) * 100


def _symmetric_mean_absolute_percentage_error(pred_values, true_values):
    # 避免除以零
    mask = true_values != 0
    if not np.any(mask):
        return np.nan
    return 200 * np.mean(np.abs(pred_values[mask] - true_values[mask]) / (np.abs(pred_values[mask]) + np.abs(true_values[mask])))

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

_EVALUATION_METRICS = (
    ("MAE", _mean_absolute_error),
    ("MSE", _mean_squared_error),
    ("MAPE", _mean_absolute_percentage_error),
    ("sMAPE", _symmetric_mean_absolute_percentage_error),
)

def save_evaluation_results(results_dict, output_dir):
    """保存评估结果到CSV文件，对多个样本的结果取平均
    
    Args:
        results_dict: 包含评估结果的字典，格式为 {sample_idx: {metric: {eval_metric: score}}}
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'evaluation_results.csv')
    
    # 获取所有指标名称
    all_metrics = set()
    for sample_results in results_dict.values():
        all_metrics.update(sample_results.keys())
    all_metrics = sorted(list(all_metrics))
    
    # 创建用于存储所有样本结果的字典
    aggregated_results = {
        metric: {
            eval_metric: [] for eval_metric, _ in _EVALUATION_METRICS
        } for metric in all_metrics
    }
    
    # 收集所有样本的结果
    for sample_results in results_dict.values():
        for metric in all_metrics:
            if metric in sample_results:
                for eval_metric, _ in _EVALUATION_METRICS:
                    if eval_metric in sample_results[metric]:
                        value = sample_results[metric][eval_metric]
                        # 只收集非nan的值
                        if not np.isnan(value):
                            aggregated_results[metric][eval_metric].append(value)
    
    # 计算平均值
    eval_metrics = [metric for metric, _ in _EVALUATION_METRICS]
    df = pd.DataFrame(index=eval_metrics, columns=all_metrics)
    
    for metric in all_metrics:
        for eval_metric in eval_metrics:
            values = aggregated_results[metric][eval_metric]
            # 如果有有效值，则计算平均值
            if values:
                df.loc[eval_metric, metric] = np.mean(values)
            else:
                df.loc[eval_metric, metric] = np.nan
    
    # 保存为CSV
    df.to_csv(output_path, encoding=ENCODING)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Moirai MoE Evaluate')

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='model/moirai-moe-1.0-R-base',
        help='Model path'
    )

    parser.add_argument(
        '--context_length',
        type=int,
        default=256,
        help='Context length'
    )

    parser.add_argument(
        '--prediction_length', '-p',
        type=int,
        default=32,
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
        '--device',
        type=str,
        default='cuda',
        help='Device'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/heating',
        help='Output dir'
    )

    parser.add_argument(
        '--forecast_method',
        type=str,
        default='median',
        choices=list(FORECAST_METHODS.keys()),
        help='预测结果处理方法'
    )

    parser.add_argument(
        '--target_dim',
        type=int,
        default=24,
        help='Target dimension'
    )

    parser.add_argument(
        '--feat_dynamic_real_dim',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--past_feat_dynamic_real_dim',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    data_list = [
        # ('/home/jiaju/dataset/heating/heating_cl512_pl32.feather', 512, 32),
        # ('/home/jiaju/dataset/heating/heating_cl512_pl64.feather', 512, 64),
        ('/home/jiaju/dataset/heating/heating_cl512_pl256.feather', 512, 256),
        ('/home/jiaju/dataset/heating/heating_cl512_pl512.feather', 512, 512)
    ]

    for idx, item in enumerate(data_list):
        data_path, context_length, prediction_length = item
        args.context_length = context_length
        args.prediction_length = prediction_length

        data = load_feather_data(
            data_path,
            args.context_length,
            args.prediction_length,
            batch_size=args.batch_size,
            device=args.device
        )

        model = MoiraiMoE(args)
        output_dir = os.path.join(args.output_dir, f'{idx}')
        os.makedirs(output_dir, exist_ok=True)
        
        final_results = {}
        for index, sample in enumerate(data):
            try:
                # 获取预测结果并处理
                preds = model.predict(sample['past'])
                if isinstance(preds, torch.Tensor):  
                    preds = preds.cpu().numpy()
                preds_df = sample_data(preds, args.forecast_method)
                
                # 获取真实值
                true_df = sample['after']
                true_np = np.array(true_df.applymap(lambda x: np.array(x, dtype=np.float32)).values.tolist())
                true_np = np.transpose(true_np, (0, 2, 1))
                batch_size, pred_len, num_metrics = true_np.shape[0], true_np.shape[1], true_np.shape[2]
                true_np = true_np.reshape(batch_size * pred_len, num_metrics)
                true_df = pd.DataFrame(true_np, columns=metrics)

                common_metrics = list(set(preds_df.columns) & set(true_df.columns))
                
                # 对每个指标分别计算评估指标
                sample_results = {}
                for metric_name in common_metrics:
                    
                    # 获取当前指标的预测值和真实值
                    pred_values = preds_df[metric_name].values
                    true_values = true_df[metric_name].values
                    
                    metric_results = {}
                    # 计算评估指标
                    for eval_metric, metric_fn in _EVALUATION_METRICS:
                        score = metric_fn(pred_values, true_values)
                        metric_results[eval_metric] = score
                    sample_results[metric_name] = metric_results
                
                final_results[index] = sample_results
            except Exception as e:
                print(f"Error in sample {index}: {e}")
                traceback.print_exc()
                break

        save_evaluation_results(final_results, output_dir)