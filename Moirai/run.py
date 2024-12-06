import numpy as np
import pandas as pd
import torch
from gluonts.dataset.split import split
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation.metrics import (
    MAE,
    MSE,
    MAPE,
    SMAPE,
    MASE
)
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from tqdm import tqdm

def evaluate_timeseries(
    data_path: str,
    model_size: str = "small",
    context_length: int = 1000,
    prediction_length: int = 96,
    patch_size: int = 32,
    batch_size: int = 32,
    num_samples: int = 100,
    test_size: int = 2000,
    windows: int = 10,
    distance: int = 96
):
    """
    评估时间序列预测模型
    
    参数:
        data_path: 数据文件路径
        model_size: 模型大小 ('small', 'base', 'large')
        context_length: 输入窗口长度
        prediction_length: 预测长度
        patch_size: 时间步长分块大小
        batch_size: 批处理大小
        num_samples: 采样数量
        test_size: 测试集大小
        windows: 评估窗口数量
        distance: 窗口间隔
    """
    
    # 1. 加载数据
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    ds = PandasDataset(dict(df))
    
    # 2. 划分测试集
    train, test_template = split(ds, offset=-test_size)
    
    # 3. 生成评估窗口
    test_data = test_template.generate_instances(
        prediction_length=prediction_length,
        windows=windows,
        distance=distance
    )
    
    # 4. 准备模型
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(
            f"Salesforce/moirai-1.0-R-{model_size}",
        ),
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    
    # 5. 创建预测器
    predictor = model.create_predictor(batch_size=batch_size)
    
    # 6. 进行预测
    print("开始预测...")
    forecasts = []
    try:
        for batch in tqdm(test_data.input):
            forecast = predictor.predict([batch])
            forecasts.extend(list(forecast))
    except torch.cuda.OutOfMemoryError:
        print(f"内存不足,减小batch_size到 {batch_size//2}")
        batch_size //= 2
        
    # 7. 计算评估指标
    print("计算评估指标...")
    metrics = {
        'mae': MAE(),
        'mse': MSE(),
        'mape': MAPE(),
        'smape': SMAPE(),
        'mase': MASE()
    }
    
    results = {}
    for name, metric in metrics.items():
        values = []
        for forecast, label in zip(forecasts, test_data.label):
            pred = forecast.samples.mean(axis=0)  # 使用预测样本的均值
            true = label['target']
            values.append(metric(pred, true))
        results[name] = np.mean(values)
    
    # 8. 输出结果
    print("\n评估结果:")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.4f}")
        
    return results

if __name__ == "__main__":
    # 示例使用
    data_path = "your_data.csv"
    results = evaluate_timeseries(
        data_path=data_path,
        model_size="small",
        context_length=1000,
        prediction_length=96,
        patch_size=32,
        batch_size=32,
        test_size=2000,
        windows=10,
        distance=96
    )