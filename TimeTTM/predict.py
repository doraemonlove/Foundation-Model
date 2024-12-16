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

# 处理输入数据
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

# 模型类
class TTMsPredictor:
    def __init__(self, args):
        self.args = args
        print("Loading model...")
        forecast_length = self.args.prediction_length
        model_path = self.args.model_path
        # 加载模型
        #需要根据预测长度，看是否需要截断或者滚更
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
            
        device = torch.device(self.args.device)
        
        self.model = model
        self.model.to(self.args.device)
        # 生成迭代器   
        temp_dir = tempfile.mkdtemp()
        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=temp_dir,
                per_device_eval_batch_size=self.args.batch_size,
                seed=self.args.seed,
                report_to="none",
            )
        )

    def predict(self,input_data):
        print("\nPerforming prediction...")
        prediction_output = self.trainer.predict(input_data)  # 获取预测输出
        predictions = prediction_output.predictions[0]

        return {
                'samples': predictions
        }
    
#保存预测结果
FORECAST_METHODS = {
    'raw_samples': lambda x: np.round(x['samples'], decimals=4),
    'mean': lambda x: np.round(x['samples'].mean(dim=1), decimals=4),
    'median': lambda x: np.round(np.median(x['samples'], axis=1), decimals=4),
    'all_stats': lambda x: {
        'samples': np.round(x['samples'], decimals=4),
        'mean': np.round(x['samples'].mean(dim=1), decimals=4),
        'median': np.round(np.median(x['samples'], axis=1), decimals=4)
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
    parser.add_argument('--batch_size', type=int, default=8, help="Batch_size,man. You no m3?")

    # 模型本身参数
    parser.add_argument('--model_path', '-m', type=str, default='ibm-granite/granite-timeseries-ttm-r2', help='Model path or name')

    # 显卡设置
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')

    # 处理输出
    parser.add_argument('--forecast_method', type=str, default='raw_samples', choices=list(FORECAST_METHODS.keys()), help='预测结果处理方法: raw_samples(原始样本), mean(均值), median(中位数), all_stats(所有统计)')
    parser.add_argument('--output_path', type=str, default='./output', help='Path to the output data file')

    args = parser.parse_args()

    if args.device.isdigit():  # 如果传入的是数字
        args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    print("\nStarting prediction pipeline...")
    print(f"Using device: {args.device}")

    # 准备输入数据
    _, _, test_data = collect_data(args)

    # 初始化模型并预测
    model = TTMsPredictor(args)
    preds = model.predict(test_data)

    # 保存预测结果
    save_data(preds, method=args.forecast_method, output_path=args.output_path)

    print("\nPrediction completed successfully!")

"""
    CUDA_VISIBLE_DEVICES=2 python predict.py \
    --data_path /home/zhupengtian/zhangqingliang/datasets/electricity.csv \
    --prediction_length 96
"""