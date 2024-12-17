import os
import argparse
import torch 
import tempfile
from transformers import Trainer, TrainingArguments, set_seed
import pandas as pd
from tsfm_public import TinyTimeMixerForPrediction, TimeSeriesPreprocessor
from tsfm_public.toolkit import get_datasets,RecursivePredictor, RecursivePredictorConfig
import taos  
import numpy as np
from tqdm import tqdm

import os
import torch
import math
import os
import tempfile
import warnings
import time
import torch
import psutil
import shutil
import subprocess  # 用于调用 nvidia-smi 获取 GPU 内存信息
import threading

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.integrations import INTEGRATION_TO_CALLBACK

from tsfm_public import TinyTimeMixerForPrediction, TrackingCallback, count_parameters, load_dataset
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions
import numpy as np


# 设置 Hugging Face镜像网址环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 自定义评估指标
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def mase(y_true, y_pred, train, m=1):
    # 计算训练集的绝对误差
    train_errors = np.abs(np.array(train[m:]) - np.array(train[:-m]))  # 滞后 m
    mae_train = np.mean(train_errors)  # 训练集的平均绝对误差
    # 计算 MASE
    return np.mean(np.abs(y_true - y_pred)) / mae_train if mae_train != 0 else np.nan

def collect_data(args,**dataset_kwargs,):

    print("\nCollecting data...")
    # Load data
    data = pd.read_csv(args.data_path, parse_dates=[args.timestamp_column])

    # Initialize TimeSeriesPreprocessor
    tsp = TimeSeriesPreprocessor(
        id_columns=args.id_columns,
        timestamp_column=args.timestamp_column,
        target_columns=args.target_columns,
        observable_columns=args.observable_columns,
        control_columns=args.control_columns,
        conditional_columns=args.conditional_columns,
        static_categorical_columns=args.static_categorical_columns,
        scaling=args.scaling,
        scaler_type=args.scaler_type,
        encode_categorical=args.encode_categorical,
        freq=args.freq,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )

    # Generate datasets
    train_dataset, valid_dataset, test_dataset = get_datasets(
        tsp,
        data,
        split_config=args.split_config,
        fewshot_fraction=args.fewshot_fraction,
        fewshot_location=args.fewshot_location,
        use_frequency_token=args.use_frequency_token,
        enable_padding=args.enable_padding,
        seed=args.seed,
        **dataset_kwargs,
    )

    return train_dataset, valid_dataset, test_dataset


def fewshot_finetune_eval(args):

    # 设定参数
    forecast_length = args.prediction_length
    model_path = args.model_path
    out_dir = args.output_path
    device = device = torch.device(args.device)
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    seed = args.seed
    num_epochs = args.epochs
    freeze_backbone=True
    

    # Get data
    train_data, valid_data, test_data = collect_data(args)

    # Load model
    print("Loading model...")
    if forecast_length == 96:
        model = TinyTimeMixerForPrediction.from_pretrained(model_path, revision="main")
    elif forecast_length < 96:
        model = TinyTimeMixerForPrediction.from_pretrained(
                model_path,
                revision="main",
                prediction_filter_length=forecast_length,
            )
    else:
        base_model = TinyTimeMixerForPrediction.from_pretrained(model_path, revision="main")
        rec_config = RecursivePredictorConfig(
            model=base_model,
            requested_prediction_length=forecast_length,
            model_prediction_length=96,
            loss=base_model.config.loss,
        )
        model = RecursivePredictor(rec_config)

    finetune_forecast_model = model.to(device) 

    

    if freeze_backbone:
        print(
            "Number of params before freezing backbone",
            count_parameters(finetune_forecast_model),
        )

        # Freeze the backbone of the model
        for param in finetune_forecast_model.backbone.parameters():
            param.requires_grad = False

        # Count params
        print(
            "Number of params after freezing the backbone",
            count_parameters(finetune_forecast_model),
        )

    # Find optimal learning rate
    # Use with caution: Set it manually if the suggested learning rate is not suitable
    if learning_rate is None:
        learning_rate, finetune_forecast_model = optimal_lr_finder(
            finetune_forecast_model,
            train_data,
            batch_size=args.batch_size,
        )
        print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

    print(f"Using learning rate = {learning_rate}")
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        report_to="none",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(out_dir, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        seed=seed,
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=1e-5,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(train_data) / (batch_size)),
    )

    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=train_data,
        eval_dataset=valid_data, 
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )
    finetune_forecast_trainer.remove_callback(INTEGRATION_TO_CALLBACK["codecarbon"])

    # 训练
    print("\nTraining...")
    finetune_forecast_trainer.train()
    # # 进行预测
    # prediction_output = finetune_forecast_trainer.predict(test_data)  # 获取预测输出
    # predictions = prediction_output.predictions[0]
    # # 获取标签
    # future_values = np.array([sample['future_values'] for sample in test_data])  

    # #计算指标
    # mse = mean_squared_error(future_values.reshape(-1), predictions.reshape(-1))
    # mae = mean_absolute_error(future_values.reshape(-1), predictions.reshape(-1))
    # mape = np.mean(np.abs((future_values - predictions) / future_values)) * 100  # 转为百分比
    # smape_value = smape(future_values, predictions)
    # mase_value = mase(future_values, predictions, future_values)  # 传入训练集（或验证集）

    # # 输出结果
    # print("均方误差 (MSE):", mse)
    # print("平均绝对误差 (MAE):", mae)
    # print("平均绝对百分比误差 (MAPE):", mape)
    # print("对称平均绝对百分比误差 (SMAPE):", smape_value)
    # print("平均绝对误差比 (MASE):", mase_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ChronosBolt Predict')

    # 处理输入数据
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument('--context_length', type=int, default=512,help="Context length for time series.")
    parser.add_argument('--prediction_length', type=int, default=96,help="Forecast length for time series.")
    parser.add_argument('--id_columns', type=str, nargs='*', default=[], help="List of ID columns.")
    parser.add_argument('--timestamp_column', type=str, default="date", help="Name of the timestamp column.")
    parser.add_argument('--target_columns', type=str, nargs='*', default=[], help="List of target columns.")
    parser.add_argument('--observable_columns', type=str, nargs='*', default=[], help="List of observable columns.")
    parser.add_argument('--control_columns', type=str, nargs='*', default=[], help="List of control columns.")
    parser.add_argument('--conditional_columns', type=str, nargs='*', default=[], help="List of conditional columns.")
    parser.add_argument('--static_categorical_columns', type=str, nargs='*', default=[], help="List of static categorical columns.")
    parser.add_argument('--freq', type=str, default="1h", help="Frequency of the time series data.")
    parser.add_argument('--scaling', type=bool, default=True, help="Whether to apply scaling.")
    parser.add_argument('--scaler_type', type=str, default="standard", help="Type of scaler to use.")
    parser.add_argument('--encode_categorical', type=bool, default=False, help="Whether to encode categorical columns.")
    parser.add_argument('--split_config', type=str, default={"train": 0.7, "test": 0.2}, help="{'train':xx 'test': xx}")
    parser.add_argument('--fewshot_fraction', type=float, default=1.0, help="Fraction of fewshot samples to use.")
    parser.add_argument('--fewshot_location', type=str, default="first", help="Location of fewshot samples.")
    parser.add_argument('--use_frequency_token', type=bool, default=False, help="Whether to use frequency token.")
    parser.add_argument('--enable_padding', type=bool, default=True, help="Whether to enable padding.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    #训练参数
    parser.add_argument('--batch_size', type=int, default=8, help="Batch_size,man. You no m3?")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument('--epochs', type=int, default=8, help="Epochs of training")

    # 模型本身参数
    parser.add_argument('--model_path', '-m', type=str, default='ibm-granite/granite-timeseries-ttm-r2', help='Model path or name')

    # 显卡设置
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')

    #输出文件夹
    parser.add_argument('--output_path', type=str, default='./output' , help='Output of finetuned model,man')

    args = parser.parse_args()

    if args.device.isdigit():  # 如果传入的是数字
        args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    print("\nStarting finetune pipeline...")
    print(f"Using device: {args.device}")

    fewshot_finetune_eval(args)

    print("\nFinetune completed successfully!")

"""
    CUDA_VISIBLE_DEVICES=2 python fewshot.py \
    --data_path /home/zhupengtian/zhangqingliang/datasets/electricity.csv \
    --prediction_length 96 \
    --epochs 8 \
    --fewshot_fraction 0.1
"""