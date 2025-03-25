import argparse
import torch
import yaml
import numpy as np
from models.UniTS_zeroshot import Model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import pandas as pd
import traceback
ENCODING = 'utf-8'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  
metrics = [
    'rpst', 'rprt', 'rpsif', 'rpsih',
    'rsst', 'rsrt', 'rssif', 'rssih',
    'acpst', 'acprt', 'acpsif', 'acpsih',
    'mrih',
    'rpvo1', 'rpvo2', 'acpvo1', 'acpvo2',
    'rpf1', 'rpf2', 'rpf3', 'acpf1', 'acpf2', 'acpf3', 'acpf4'
]

def read_task_data_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    task_dataset_config = config.get('task_dataset', {})
    return task_dataset_config


def get_task_data_config_list(task_data_config, default_batch_size=None):
    task_data_config_list = []

    for task_name, task_config in task_data_config.items():
        task_config['max_batch'] = default_batch_size
        task_data_config_list.append([task_name, task_config])

    return task_data_config_list

class UniTS_Zeroshot:
    def __init__(self, args, task_data_config_list):
        self.args = args
        self.task_data_config_list = task_data_config_list
        self.model = Model(args, task_data_config_list)

        if os.path.exists(self.args.pretrained_weight):
            pretrain_weight_path = self.args.pretrained_weight
            print('loading pretrained model:', pretrain_weight_path)
            if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                state_dict = torch.load(
                    pretrain_weight_path, map_location='cpu')['student']
                ckpt = {}
                for k, v in state_dict.items():
                    if not ('cls_prompts' in k):
                        ckpt[k] = v
            else:
                ckpt = torch.load(pretrain_weight_path, map_location='cpu')
            msg = self.model.load_state_dict(ckpt, strict=False)
            print(msg)
        else:
            print("no ckpt found!")
            exit()

    def predict(self, batch_x, task_id):
        pred_len = self.task_data_config_list[task_id][1]['pred_len']
        self.model.to(self.args.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch_x, None, task_id=task_id, task_name='long_term_forecast')
            outputs = outputs[:, -pred_len:, :]
            outputs = outputs.detach().cpu()
        return outputs
    
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

    # 如果存在 'rpsih' 和 'acpsih'，添加 'mrih'
    if 'rpsih' in df.columns and 'acpsih' in df.columns:
        df['mrih'] = df['rpsih'] + df['acpsih']
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

        # DataFrame 格式的 `before` 和 `after`
        before_df = batch_df.applymap(lambda x: x[:context_length])
        after_df = batch_df.applymap(lambda x: x[context_length: context_length + prediction_length])

        before_np = np.array(before_df.applymap(lambda x: np.array(x, dtype=np.float32)).values.tolist())
        before_np = np.transpose(before_np, (0, 2, 1))

        # 将 `after` 转换为 DataFrame   
        after_np = np.array(after_df.applymap(lambda x: np.array(x, dtype=np.float32)).values.tolist())
        after_np = np.transpose(after_np, (0, 2, 1))

        data.append({
            "past": before_values,  # 供模型使用的历史数据
            "before": before_np,  # DataFrame 格式的历史数据
            "after": after_np,  # DataFrame 格式的真实值
        })
    
    print(f'>>> Loaded {len(data)} batches of size {batch_size}')
    return data


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

def plot_metrics_to_pdf(before_np: np.ndarray, pred_np: np.ndarray, gt_np: np.ndarray, pdf_path: str):
    """为每个指标创建历史数据和预测对比图，同一张图中前半段显示历史数据，后半段显示预测值和真实值"""
    
    with PdfPages(pdf_path) as pdf:
        for idx, metric in enumerate(metrics):
            fig, ax = plt.subplots(figsize=(15, 6))
            
            # 获取时间点数量
            before_len = before_np.shape[0]
            pred_len = pred_np.shape[0]
            
            # 绘制历史数据（前半部分）
            ax.plot(range(before_len), before_np[:, idx], 
                   label='历史数据', color='gray')
            
            # 绘制预测值和真实值（后半部分）
            x_pred = range(before_len, before_len + pred_len)
            ax.plot(x_pred, pred_np[:, idx], 
                   label='预测值', color='red')
            ax.plot(x_pred, gt_np[:, idx], 
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniTS supervised training')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='ALL_task',
                        help='task name')    
    parser.add_argument('--is_training', type=int,
                        required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True,
                        default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='UniTS',
                        help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=False,
                        default='All', help='dataset type')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--task_data_config_path', type=str,
                        default='exp/all_task.yaml', help='root path of the task and data yaml file')
    parser.add_argument('--subsample_pct', type=float,
                        default=None, help='subsample percent')

    # ddp
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loader num workers')
    parser.add_argument("--memory_check", action="store_true", default=True)
    parser.add_argument("--large_model", action="store_true", default=True)

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int,
                        default=10, help='train epochs')
    parser.add_argument("--prompt_tune_epoch", type=int, default=0)
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--acc_it', type=int, default=1,
                        help='acc iteration to enlarge batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='optimizer learning rate')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='optimizer min learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='optimizer weight decay')
    parser.add_argument('--layer_decay', type=float,
                        default=None, help='optimizer layer decay')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--lradj', type=str,
                        default='supervised', help='adjust learning rate')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='save location of model checkpoints')
    parser.add_argument('--pretrained_weight', type=str, default=None,
                        help='location of pretrained model checkpoints')
    parser.add_argument('--debug', type=str,
                        default='enabled', help='disabled')
    parser.add_argument('--project_name', type=str,
                        default='tsfm-multitask', help='wandb project name')

    # model settings
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument("--share_embedding",
                        action="store_true", default=False)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--prompt_num", type=int, default=5)
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')

    # task related settings
    # forecasting task
    parser.add_argument('--inverse', action='store_true',
                        help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float,
                        default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float,
                        default=1.0, help='prior anomaly ratio (%)')

    # zero-shot-forecast-new-length
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_offset", type=int, default=0)
    parser.add_argument('--zero_shot_forecasting_new_length',
                        type=str, default=None, help='unify')

    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    task_data_config = read_task_data_config(args.task_data_config_path)
    task_data_config_list = get_task_data_config_list(task_data_config)

    data_list = [
        ('/home/jiaju/dataset/heating/heating_cl512_pl32.feather', 512, 32),
        ('/home/jiaju/dataset/heating/heating_cl512_pl64.feather', 512, 64),
        ('/home/jiaju/dataset/heating/heating_cl512_pl256.feather', 512, 256),
        ('/home/jiaju/dataset/heating/heating_cl512_pl512.feather', 512, 512)
    ]

    units = UniTS_Zeroshot(args, task_data_config_list)

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

        output_dir = os.path.join(args.output_dir, f'{idx}')
        os.makedirs(output_dir, exist_ok=True)
        
        final_results = {}
        for index, sample in enumerate(data):
            try:
                # 获取预测结果并处理
                preds = units.predict(sample['past'], task_id=idx)
                if isinstance(preds, torch.Tensor):  
                    preds = preds.cpu().numpy()

                # 获取真实值
                true_np = sample['after']
                
                if index % 10 == 0:
                    plot_metrics_to_pdf(sample['before'][0], preds[0], true_np[0], os.path.join(output_dir, f'{index}.pdf'))

                # 对每个指标分别计算评估指标
                sample_results = {}
                for metric_idx, metric_name in enumerate(metrics):
                    
                    # 获取当前指标的预测值和真实值
                    pred_values = preds[:, metric_idx, :]
                    true_values = true_np[:, metric_idx, :]
                    
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




    