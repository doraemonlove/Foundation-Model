import argparse
import logging
from tqdm import tqdm
import taos
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

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

class TimeMoE:
    def __init__(self, model_path, device, prediction_length, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            # attn_implementation='flash_attention_2',
            torch_dtype='auto',
            trust_remote_code=True,
        )
        logging.info(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')

        self.model = model
        self.device = device
        self.prediction_length = prediction_length
        self.model.eval()

    def predict(self, input_data):
        model = self.model
        device = self.device
        prediction_length = self.prediction_length

        # 确保输入是2D张量 [1, seq_length]
        if len(input_data.shape) == 1:
            inputs = input_data.unsqueeze(0)
        else:
            inputs = input_data
            
        # 标准化输入数据
        mean = inputs.mean(dim=-1, keepdim=True)
        std = inputs.std(dim=-1, keepdim=True)
        normed_inputs = (inputs - mean) / std
            
        outputs = model.generate(
            inputs=normed_inputs.to(device).to(model.dtype),
            max_new_tokens=prediction_length,
        )
        
        # 只取预测部分并反标准化
        normed_preds = outputs[:, -prediction_length:].squeeze(0)
        preds = normed_preds * std + mean
        return preds


def collect_data(start_time=None, end_time=None):
    data_path = 'dataset/electricity/electricity.csv'
    target = 'OT'
    df = pd.read_csv(data_path)
    target_values = df[[target]].values.T
    target_values = torch.tensor(target_values[:,-256:])
    return target_values

def save_data(preds, output_path=None):
    print('>>> Preds: ', preds)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TimeMoE Predict')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='./TimeMoE-200M',
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
        '--prediction_length', '-p',
        type=int,
        default=24,
        help='Prediction length'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path'
    )
    args = parser.parse_args()

    device = 'cpu'

    input_data = collect_data(args.start_time, args.end_time)

    model = TimeMoE(
        args.model,
        device,
        prediction_length=args.prediction_length
    )

    preds = model.predict(input_data)

    save_data(preds, args.output_path)
