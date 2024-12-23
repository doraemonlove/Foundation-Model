import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from data_provider.data_factory import data_provider
from utils.dataloader import BalancedDataLoaderIterator
from utils.ddp import  is_main_process, gather_tensors_from_all_gpus, init_distributed_mode
from tqdm import tqdm
import yaml
import copy

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


def change_config_list_pred_len(task_data_config_list, task_data_config, offset):
    print("Warning: change the forecasting len and remove the cls task!")
    new_task_data_config = copy.deepcopy(task_data_config)
    new_task_data_config_list = copy.deepcopy(task_data_config_list)
    for task_name, task_config in new_task_data_config.items():
        if task_config['task_name'] == 'long_term_forecast':
            new_task_data_config[task_name]['pred_len'] += offset
        else:
            del new_task_data_config[task_name]

    for each_config in new_task_data_config_list:
        if each_config[1]['task_name'] == 'long_term_forecast':
            each_config[1]['pred_len'] += offset
        else:
            del each_config

    return new_task_data_config_list, new_task_data_config


def init_and_merge_datasets(data_loader_list):
    dataloader = BalancedDataLoaderIterator(data_loader_list)
    train_steps = dataloader.__len__()
    return dataloader, train_steps

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

class UnitsPredictor(object):
    def __init__(self, args):
        super(UnitsPredictor, self).__init__()

        self.args = args
        self.ori_task_data_config = read_task_data_config(
            self.args.task_data_config_path)
        self.ori_task_data_config_list = get_task_data_config_list(
            self.ori_task_data_config, default_batch_size=self.args.batch_size)

        if self.args.zero_shot_forecasting_new_length is not None:
            print("Change the forecasting len!")
            self.task_data_config_list, self.task_data_config = change_config_list_pred_len(self.ori_task_data_config_list, self.ori_task_data_config, self.args.offset)
        else:
            self.task_data_config = self.ori_task_data_config
            self.task_data_config_list = self.ori_task_data_config_list
        device_id = dist.get_rank() % torch.cuda.device_count()
        self.device_id = device_id
        print("device id", self.device_id)
        self.model = self._build_model()

    def _build_model(self, ddp=True):
        import importlib
        module = importlib.import_module("models." + self.args.model)
        model = module.Model(
            self.args, self.task_data_config_list).to(self.device_id)
        if ddp:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device_id],
                                                        find_unused_parameters=True, gradient_as_bucket_view=True, static_graph=False)
        return model

    def _get_data(self, flag, test_anomaly_detection=False):
        if self.args.zero_shot_forecasting_new_length is not None:
            _, max_offset_task_data_config = change_config_list_pred_len(self.ori_task_data_config_list, self.ori_task_data_config, self.args.max_offset)
            this_task_data_config = max_offset_task_data_config
        else:
            this_task_data_config = self.task_data_config

        data_set_list = []
        data_loader_list = []

        for task_data_name, task_config in this_task_data_config.items():
            if task_config['task_name'] == 'classification' and flag == 'val':
                # TODO strange that no val set is used for classification. Set to test set for val
                flag = 'test'
            if test_anomaly_detection and task_config['task_name'] == 'anomaly_detection':
                train_data_set, train_data_loader = data_provider(
                    self.args, task_config, flag='train', ddp=False)
                data_set, data_loader = data_provider(
                    self.args, task_config, flag, ddp=False)  # ddp false to avoid shuffle
                data_set_list.append([train_data_set, data_set])
                data_loader_list.append([train_data_loader, data_loader])
                print(task_data_name, len(data_set))
            else:
                data_set, data_loader = data_provider(
                    self.args, task_config, flag, ddp=True)
                data_set_list.append(data_set)
                data_loader_list.append(data_loader)
                print(task_data_name, len(data_set))
        return data_set_list, data_loader_list

    def predict(self, setting, load_pretrain=False, test_data_list=None, test_loader_list=None):
        self.path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(self.path) and is_main_process():
            os.makedirs(self.path)
        if test_data_list is None or test_loader_list is None:
            test_data_list, test_loader_list = self._get_data(
                flag='test', test_anomaly_detection=True)
        if load_pretrain:
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

        preds = []

        for task_id, (test_data, test_loader) in enumerate(zip(test_data_list, test_loader_list)):
            task_name = self.task_data_config_list[task_id][1]['task_name']
            data_task_name = self.task_data_config_list[task_id][0]
            if task_name == 'long_term_forecast':
                if self.args.zero_shot_forecasting_new_length == 'unify':
                    preds = self.long_term_forecast_offset_unify(
                        setting, test_data, test_loader, data_task_name, task_id)
                else:
                    preds = self.long_term_forecast(
                        setting, test_data, test_loader, data_task_name, task_id)

        return preds  # 只返回预测结果

    def long_term_forecast(self, setting, test_data, test_loader, data_task_name, task_id):
        config = self.task_data_config_list[task_id][1]
        label_len = config['label_len']
        pred_len = config['pred_len']
        features = config['features']

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, _, _) in enumerate(tqdm(test_loader, desc="Processing batches")):
                batch_x = batch_x.float().to(self.device_id)

                dec_inp = None
                batch_x_mark = None
                batch_y_mark = None

                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, task_id=task_id, task_name='long_term_forecast')

                f_dim = -1 if features == 'MS' else 0
                outputs = outputs[:, -pred_len:, f_dim:]

                outputs = outputs.detach().cpu()
                if test_data.scale and self.args.inverse:
                    outputs = test_data.inverse_transform(outputs)

                preds.append(outputs)

        preds = gather_tensors_from_all_gpus(preds, self.device_id)
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        return preds  # 只返回预测结果

    def long_term_forecast_offset_unify(self, setting, test_data, test_loader, data_task_name, task_id):
        config = self.task_data_config_list[task_id][1]
        pred_len = config['pred_len']
        features = config['features']
        max_pred_len = pred_len - self.args.offset + self.args.max_offset

        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device_id)
                batch_y = batch_y.float().to(self.device_id)
                batch_y = batch_y[:, -max_pred_len:][:, :pred_len]

                dec_inp = None
                batch_x_mark = None
                batch_y_mark = None

                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, task_id=task_id, task_name='long_term_forecast')

                f_dim = -1 if features == 'MS' else 0
                outputs = outputs[:, -pred_len:, f_dim:]

                outputs = outputs.detach().cpu()
                if test_data.scale and self.args.inverse:
                    outputs = test_data.inverse_transform(outputs)

                preds.append(outputs)

        preds = gather_tensors_from_all_gpus(preds, self.device_id)
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        return preds  # 只返回预测结果
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniTS predict')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name')
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--model_id', type=str, default='predict', help='model id')
    parser.add_argument('--model', type=str, default='UniTS', help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=False,
                        default='custom', help='dataset type')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--task_data_config_path', type=str,
                        default='exp/all_task.yaml', help='root path of the task and data yaml file')

    # optimization
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='save location of model checkpoints')
    parser.add_argument('--pretrained_weight', type=str, default=None,
                        help='location of pretrained model checkpoints')
    parser.add_argument('--debug', type=str,
                        default='disabled', help='enabled')
    parser.add_argument('--project_name', type=str,
                        default='tsfm-multitask', help='wandb project name')

    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--subsample_pct', type=float, default=None, help='subsample percent')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

    parser.add_argument('--des', type=str, default='test', help='experiment description')

    # model settings
    parser.add_argument('--d_model', type=int, default=64,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3,
                        help='num of encoder layers')
    parser.add_argument("--share_embedding",
                        action="store_true", default=False)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--prompt_num", type=int, default=10)
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')

    # forecasting task
    parser.add_argument('--inverse', action='store_true',
                        help='inverse output data', default=False)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_offset", type=int, default=0)
    parser.add_argument('--zero_shot_forecasting_new_length',
                        type=str, default=None, help='unify')
    parser.add_argument('--output_path', type=str, default='./output', help='Path to the output data file')
    parser.add_argument('--save_method', type=str, default='raw_samples', help='Method to save predictions (options: raw_samples, mean, median, all_stats)')
    args = parser.parse_args()
    
    # 初始化分布式训练
    init_distributed_mode(args)  # 确保在创建 UnitsPredictor 之前调用

    if args.fix_seed is not None:
        random.seed(args.fix_seed)
        torch.manual_seed(args.fix_seed)
        np.random.seed(args.fix_seed)

    predictor = UnitsPredictor(args)
    ii = 0
    setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.d_model,
        args.e_layers,
        args.des, ii)
    res = predictor.predict(setting, load_pretrain=True)
    print(res)
    save_data({'samples':res}, method=args.save_method, output_path=args.output_path)
    torch.cuda.empty_cache()

"""
CUDA_VISIBLE_DEVICES=6 torchrun --nnodes 1 --nproc_per_node 1 predict.py \
  --pretrained_weight /models/UniTS/units_x64_supervised_checkpoint.pth \
  --task_data_config_path ./data_provider/zeroshot_task.yaml\
  --batch_size 64\
  --output_path ./output
"""
