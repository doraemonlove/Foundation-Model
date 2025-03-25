# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import logging
from collections import ChainMap
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union
import torch
import numpy as np
import pandas as pd
from gluonts.dataset import DataEntry
from gluonts.dataset.split import TestData
from gluonts.ev.ts_stats import seasonal_error
from gluonts.itertools import batcher, prod
from gluonts.model import Forecast, Predictor
from gluonts.time_feature import get_seasonality
from toolz import first, valmap
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  

@dataclass
class BatchForecast:
    """
    Wrapper around ``Forecast`` objects, that adds a batch dimension
    to arrays returned by ``__getitem__``, for compatibility with
    ``gluonts.ev``.
    """

    forecasts: List[Forecast]
    allow_nan: bool = False

    def __getitem__(self, name):
        values = [forecast[name].T for forecast in self.forecasts]
        res = np.stack(values, axis=0)

        if np.isnan(res).any():
            if not self.allow_nan:
                raise ValueError("Forecast contains NaN values")

            logger.warning("Forecast contains NaN values. Metrics may be incorrect.")

        return res


def _get_data_batch(
    input_batch: List[DataEntry],
    label_batch: List[DataEntry],
    forecast_batch: List[Forecast],
    seasonality: Optional[int] = None,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
) -> ChainMap:
    label_target = np.stack([label["target"] for label in label_batch], axis=0)
    if mask_invalid_label:
        label_target = np.ma.masked_invalid(label_target)

    other_data = {
        "label": label_target,
    }

    seasonal_error_values = []
    for input_ in input_batch:
        seasonality_entry = seasonality
        if seasonality_entry is None:
            seasonality_entry = get_seasonality(input_["start"].freqstr)
        input_target = input_["target"]
        if mask_invalid_label:
            input_target = np.ma.masked_invalid(input_target)
        seasonal_error_values.append(
            seasonal_error(
                input_target,
                seasonality=seasonality_entry,
                time_axis=-1,
            )
        )
    other_data["seasonal_error"] = np.array(seasonal_error_values)

    return ChainMap(
        other_data, BatchForecast(forecast_batch, allow_nan=allow_nan_forecast)  # type: ignore
    )


def _maximum_error(pred_values, true_values):
    return np.max(np.abs(pred_values - true_values))


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


def plot_prediction_vs_truth(history_values, pred_values, true_values, metric_idx, sample_idx, output_dir):
    """绘制历史值、预测值与真实值的对比图
    Args:
        history_values: 历史值序列
        pred_values: 预测值序列
        true_values: 真实值序列
        metric_idx: 指标索引
        sample_idx: 样本索引
        output_dir: 图片保存目录
    """
    plt.figure(figsize=(15, 7))
    
    # 计算时间轴
    history_len = len(history_values)
    pred_len = len(pred_values)
    total_len = history_len + pred_len
    
    # 绘制历史值
    plt.plot(range(history_len), history_values, 
            label='历史值', color='gray', alpha=0.5)
    
    # 绘制预测值和真实值
    plt.plot(range(history_len, total_len), true_values, 
            label='真实值', color='blue')
    plt.plot(range(history_len, total_len), pred_values, 
            label='预测值', color='red')
    
    # 添加垂直分隔线标示历史值和预测值的分界
    plt.axvline(x=history_len-1, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f'指标 {metric_idx} - 样本 {sample_idx} 的预测对比')
    plt.legend()
    plt.grid(True)
    
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / f'prediction_metric_{metric_idx}_sample_{sample_idx}.png')
    plt.close()


def evaluate_forecasts_raw(
    forecasts: Iterable[Forecast],
    *,
    test_data: TestData,
    metrics,
    axis: Optional[Union[int, tuple]] = None,
    batch_size: int = 100,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    seasonality: Optional[int] = None,
    plot_interval: int = 10,
    plot_output_dir: Optional[str] = None,
) -> dict:
    if plot_output_dir is None:
        try:
            hydra_output_dir = HydraConfig.get().runtime.output_dir
            plot_output_dir = f"{hydra_output_dir}/prediction_plots"
        except:
            plot_output_dir = "prediction_plots"

    label_ndim = first(test_data.label)["target"].ndim

    assert label_ndim in [1, 2]

    if axis is None:
        axis = tuple(range(label_ndim + 1))
    if isinstance(axis, int):
        axis = (axis,)

    assert all(ax in range(3) for ax in axis)

    # 初始化收集所有预测值和真实值的字典
    all_predictions = {}
    all_true_values = {}
    
    input_batches = batcher(test_data.input, batch_size=batch_size)
    label_batches = batcher(test_data.label, batch_size=batch_size)
    forecast_batches = batcher(forecasts, batch_size=batch_size)

    batch_count = 0
    pbar = tqdm()
    for input_batch, label_batch, forecast_batch in zip(
        input_batches, label_batches, forecast_batches
    ):
        for i, (label, forecast) in enumerate(zip(label_batch, forecast_batch)):
            samples = forecast.samples
            pred_mean = np.mean(samples, axis=0)
            label_target = label["target"].T
            input_target = input_batch[i]["target"].T  # 获取历史值
            
            # 判断是单指标还是多指标
            if len(label_target.shape) == 1 or label_target.shape[1] == 1:
                # 单指标情况
                if batch_count % plot_interval == 0 and i == 0:
                    plot_prediction_vs_truth(
                        history_values=input_target[-256:] if len(input_target.shape) == 1 else input_target[-256:, 0],
                        pred_values=pred_mean if len(pred_mean.shape) == 1 else pred_mean[:, 0],
                        true_values=label_target if len(label_target.shape) == 1 else label_target[:, 0],
                        metric_idx=0,
                        sample_idx=batch_count,
                        output_dir=plot_output_dir
                    )
                
                metric_name = "metric_0"
                if metric_name not in all_predictions:
                    all_predictions[metric_name] = []
                    all_true_values[metric_name] = []
                
                all_predictions[metric_name].append(pred_mean if len(pred_mean.shape) == 1 else pred_mean[:, 0])
                all_true_values[metric_name].append(label_target if len(label_target.shape) == 1 else label_target[:, 0])
            else:
                # 多指标情况（原有代码）
                if batch_count % plot_interval == 0 and i == 0:
                    for metric_idx in range(label_target.shape[1]):
                        plot_prediction_vs_truth(
                            history_values=input_target[-256:, metric_idx],
                            pred_values=pred_mean[:, metric_idx],
                            true_values=label_target[:, metric_idx],
                            metric_idx=metric_idx,
                            sample_idx=batch_count,
                            output_dir=plot_output_dir
                        )
                
                for metric_idx in range(label_target.shape[1]):
                    metric_name = f"metric_{metric_idx}"
                    if metric_name not in all_predictions:
                        all_predictions[metric_name] = []
                        all_true_values[metric_name] = []
                    
                    all_predictions[metric_name].append(pred_mean[:, metric_idx])
                    all_true_values[metric_name].append(label_target[:, metric_idx])

        pbar.update(len(forecast_batch))
    pbar.close()

    # 计算最终的评估指标
    final_results = {}
    for metric_name in all_predictions.keys():
        pred_values = np.concatenate(all_predictions[metric_name])
        true_values = np.concatenate(all_true_values[metric_name])
        
        final_results[metric_name] = {
            "MAE": _mean_absolute_error(pred_values, true_values),
            "MSE": _mean_squared_error(pred_values, true_values),
            "MAPE": _mean_absolute_percentage_error(pred_values, true_values),
            "sMAPE": _symmetric_mean_absolute_percentage_error(pred_values, true_values)
        }

    return final_results


def evaluate_forecasts(
    forecasts: Iterable[Forecast],
    *,
    test_data: TestData,
    metrics,
    axis: Optional[Union[int, tuple]] = None,
    batch_size: int = 100,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    seasonality: Optional[int] = None,
) -> pd.DataFrame:
    """
    Evaluate ``forecasts`` by comparing them with ``test_data``, according
    to ``metrics``.

    .. note:: This feature is experimental and may be subject to changes.

    The optional ``axis`` arguments controls aggregation of the metrics:
    - ``None`` (default) aggregates across all dimensions
    - ``0`` aggregates across the dataset
    - ``1`` aggregates across the first data dimension (time, in the univariate setting)
    - ``2`` aggregates across the second data dimension (time, in the multivariate setting)

    Return results as a Pandas ``DataFrame``.
    """
    metrics_values = evaluate_forecasts_raw(
        forecasts=forecasts,
        test_data=test_data,
        metrics=metrics,
        axis=axis,
        batch_size=batch_size,
        mask_invalid_label=mask_invalid_label,
        allow_nan_forecast=allow_nan_forecast,
        seasonality=seasonality,
    )

    # 修改DataFrame创建部分
    df_data = {}
    for metric_name, metric_dict in metrics_values.items():
        for eval_name, value in metric_dict.items():
            col_name = f"{metric_name}_{eval_name}"
            df_data[col_name] = [value]  # 将scalar值包装在列表中

    return pd.DataFrame(df_data)


def evaluate_model(
    model: Predictor,
    *,
    test_data: TestData,
    metrics,
    axis: Optional[Union[int, tuple]] = None,
    batch_size: int = 100,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    seasonality: Optional[int] = None,
) -> pd.DataFrame:
    """
    Evaluate ``model`` when applied to ``test_data``, according
    to ``metrics``.

    .. note:: This feature is experimental and may be subject to changes.

    The optional ``axis`` arguments controls aggregation of the metrics:
    - ``None`` (default) aggregates across all dimensions
    - ``0`` aggregates across the dataset
    - ``1`` aggregates across the first data dimension (time, in the univariate setting)
    - ``2`` aggregates across the second data dimension (time, in the multivariate setting)

    Return results as a Pandas ``DataFrame``.
    """
    forecasts = model.predict(test_data.input)

    return evaluate_forecasts(
        forecasts=forecasts,
        test_data=test_data,
        metrics=metrics,
        axis=axis,
        batch_size=batch_size,
        mask_invalid_label=mask_invalid_label,
        allow_nan_forecast=allow_nan_forecast,
        seasonality=seasonality,
    )
