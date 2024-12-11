import torch
import pandas as pd
import argparse
import taos
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
import numpy as np

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

class MoiraiMoE:
    def __init__(self, args):
        self.args = args
        self.model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(self.args.model),
            prediction_length=self.args.prediction_length,
            context_length=self.args.context_length,
            patch_size=self.args.patch_size,
            num_samples=self.args.num_samples,
            target_dim=1,
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

def save_data(preds, method='median', output_path=None):
    """
    处理并保存预测结果
    
    Args:
        preds: 预测结果
        method: 结果处理方法，可选 'median', 'mean'
        output_path: 输出路径
    """
    if method not in FORECAST_METHODS:
        raise ValueError(f"不支持的方法: {method}. 可用方法: {list(FORECAST_METHODS.keys())}")
    
    processed_preds = FORECAST_METHODS[method](preds[0])
    print(f'>>> Processed Preds ({method}): ', processed_preds)
    
    # if output_path:
    #     np.save(output_path, processed_preds)
    
    return processed_preds

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
        '--output_path',
        type=str,
        default=None,
        help='Output path'
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

    input_data = collect_data(args.start_time, args.end_time, args.context_length)
    model = MoiraiMoE(args)

    preds = model.predict(input_data)
    save_data(preds, method=args.forecast_method, output_path=args.output_path)
