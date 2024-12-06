import argparse
import logging
from tqdm import tqdm
import taos
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from models import Timer


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

class Model:
    def __init__(self, args):
        self.args = args
        self.model = Timer.Model(args).to(args.device)
        if args.use_finetune_model:
            self.model.load_state_dict(torch.load(args.checkpoints))

    def predict(self, input_data):
        
        # 如果需要进行数据标准化
        if self.args.scale:
            scaler = StandardScaler()
            train_data = input_data.reshape(-1, input_data.shape[-1])
            scaler.fit(train_data)
            input_data = torch.from_numpy(scaler.transform(train_data).reshape(input_data.shape)).float().to(self.args.device)

        inference_steps = self.args.output_len // self.args.pred_len
        dis = self.args.output_len - inference_steps * self.args.pred_len
        if dis != 0:
            inference_steps += 1
        pred_y = []
        for j in range(inference_steps):
            if len(pred_y) != 0:
                input_data = torch.cat([input_data[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
            
            outputs = self.model(input_data)
            f_dim = -1 if self.args.features == 'MS' else 0
            pred_y.append(outputs[:, -self.args.pred_len:, :])
        pred_y = torch.cat(pred_y, dim=1)
        if dis != 0:
            pred_y = pred_y[:, :-dis, :]

        outputs = pred_y.detach().cpu()
        outputs = outputs[:, :, f_dim:]
        
        # 如果需要进行反标准化
        if self.args.scale:
            outputs = scaler.inverse_transform(outputs.reshape(-1, outputs.shape[-1])).reshape(outputs.shape)

        return outputs

def collect_data(start_time=None, end_time=None, seq_len=None, target=None):
    data_path = 'dataset/electricity/electricity.csv'
    df = pd.read_csv(data_path)
    target_values = df[[target]].values.T
    target_values = torch.tensor(target_values[:,-seq_len:])
    return target_values.unsqueeze(-1)

def save_data(preds, output_path=None):
    print('>>> Preds: ', preds.shape)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Timer Predict')
    
    # basic config
    parser.add_argument('--task_name', type=str, default='forecast',
                        help='task name, options:[forecast, imputation, anomaly_detection]')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/forecast_electricity_sr_1_Timer_custom_ftM_sl480_ll384_pl96_pl96_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp/checkpoint.pth', help='location of model checkpoints')
    parser.add_argument('--scale', action='store_true', help='scale output data', default=True)

    # model define
    parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=8, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/Timer_forecast_1.0.ckpt', help='ckpt file')

    parser.add_argument('--patch_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--subset_rand_ratio', type=float, default=1, help='mask ratio')

    # autoregressive configs
    parser.add_argument('--use_ims', action='store_true', help='Iterated multi-step', default=True)
    parser.add_argument('--output_len', type=int, default=96, help='output len')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=480, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=16, help='prediction sequence length')
    parser.add_argument('--use_finetune_model', type=int, default=0, help='use finetune model')
    parser.add_argument('--is_finetuning', type=int, default=0, help='is finetuning')

    # data
    parser.add_argument('--start_time', type=int, default=0, help='start time')
    parser.add_argument('--end_time', type=int, default=0, help='end time')

    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--output_path', type=str, default='./preds.csv', help='output path')
    
    args = parser.parse_args()

    input_data = collect_data(args.start_time, args.end_time, args.seq_len, args.target)

    model = Model(args)

    preds = model.predict(input_data)

    save_data(preds, args.output_path)
