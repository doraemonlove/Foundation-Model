from collections import defaultdict
import csv
from datetime import timedelta
import json
import os
import pickle
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
import yaml

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from util.util import get_f1

# SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
# PDT = 20  # prediction length: any positive integer
# CTX = 200  # context length: any positive integer
# PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
# BSZ = 32  # batch size: any positive integer
# TEST = 100  # test set length: any positive integer

# TEST = 5100  # test set length: any positive integer


# Read data into pandas DataFrame
# url = (
#     "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
#     "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
# )
# df0 = pd.read_csv(url, index_col=0, parse_dates=True)
# print(df0)

if __name__ == '__main__':
    all_results = []
    ratios = np.arange(0,100,0.1)
    all_results_ratio = []
    root_path = "dataset/HVAC"
    SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
    PDT = 20  # prediction length: any positive integer
    CTX = 200  # context length: any positive integer
    PSZ = 32  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
    BSZ = 32  # batch size: any positive integer
    using_finetuned_model = True

    ratios = np.arange(0,10,0.1)
    all_best_f1s = []
    all_best_f1s_ratio = []
    all_results_ratio = []
    saved_middle_var = {}
    for fault_class in [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]:
        some_fault_path = os.path.join(root_path, fault_class)
        for degree in [d for d in os.listdir(some_fault_path) if os.path.isdir(os.path.join(some_fault_path, d))]:
            label_path = os.path.join(some_fault_path, degree, f'{degree}_label.csv')
            with open(label_path, 'r') as file:
                csv_reader = csv.reader(file, delimiter=',')
                label = np.array([row[1] for row in csv_reader][1:], dtype=np.float32)
            data_path = os.path.join(some_fault_path, degree, f'{degree}.csv')
            # url = "./dataset/HVAC/cf/cf6/cf6.csv"
            ground_truth = pd.read_csv(data_path)
            start_time = pd.Timestamp('2024-01-01 00:00:00')
            time_series = [start_time + timedelta(seconds=60 * i) for i in range(len(ground_truth))]
            ground_truth['timestamp'] = time_series
            ground_truth.set_index('timestamp', inplace=True)
            ground_truth.index.name = None
            # print(time_series)
            # print(df)
            TEST = ((ground_truth.shape[0] - CTX) // PDT) * PDT # test set length: any positive integer (4980)
            # Convert into GluonTS dataset
            ds = PandasDataset(dict(ground_truth))
            # Split into train/test set
            train, test_template = split(
                ds, offset=-TEST
            )  # assign last TEST time steps as test set

            # Construct rolling window evaluation
            test_data = test_template.generate_instances(
                prediction_length=PDT,  # number of time steps for each prediction
                windows=TEST // PDT,  # number of windows in rolling window evaluation
                distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
            )

            # Prepare pre-trained model by downloading model weights from huggingface hub
            if using_finetuned_model:
                # 加载微调的模型检查点
                print('Loading finetuned model: epoch=10-step=1100.ckpt')
                model = MoiraiForecast.load_from_checkpoint(
                    './checkpoints/moirai-small-finetuned-hvac/epoch=10-step=1100.ckpt',  # 微调检查点的路径
                    module=MoiraiModule.from_pretrained(f"./checkpoints/moirai-1.0-R-{SIZE}"),
                    prediction_length=PDT,
                    context_length=CTX,
                    patch_size=PSZ,
                    num_samples=100,
                    target_dim=1,
                    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
                )
            else:
                print('Loading pretrained model: ', f"./checkpoints/moirai-1.0-R-{SIZE}")
                model = MoiraiForecast(
                    module=MoiraiModule.from_pretrained(f"./checkpoints/moirai-1.0-R-{SIZE}"),
                    prediction_length=PDT,
                    context_length=CTX,
                    patch_size=PSZ,
                    num_samples=100,
                    target_dim=1,
                    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
                )
            model.eval()
            print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.')
            predictor = model.create_predictor(batch_size=BSZ)
            forecasts = predictor.predict(test_data.input)
            forecasts = list(forecasts)
            test = True
            output = {}
            for item in forecasts:
                item_id = item.item_id
                if item_id not in output:
                    output[item_id] = []
                output[item_id].extend(item.mean)
            df_input = ground_truth[CTX:CTX + TEST]
            label = label[CTX:CTX + TEST]
            df_output = pd.DataFrame({k: v for k, v in output.items()})
            df_output.index = df_input.index
            print('len(label): ', len(label))
            print('len(df_input): ', len(df_input))
            print('len(df_output): ', len(df_output))
            scaler = StandardScaler()
            df_input_fit = scaler.fit_transform(df_input)
            df_output_fit = scaler.transform(df_output)
            score = np.mean((df_input_fit - df_output_fit) ** 2, axis=1)
            saved_middle_var[degree] = {'df_input': df_input, 'df_output': df_output, 'score': score, 'label': label}
            # 输出每行的损失
            print('len(score): ', len(score))
            now_best_f1 = -1
            now_best_f1_ratio = -1
            for ratio in ratios:
                threshold = np.percentile(score, 100 - ratio)
                (f1,precision,recall), (predict, actual) = get_f1(score, label, threshold)
                if f1 > now_best_f1:
                    now_best_f1 = f1
                    now_best_f1_ratio = ratio
                all_results_ratio.append({'ratio': ratio, 'f1': f1})
            all_best_f1s.append(now_best_f1)
            all_best_f1s_ratio.append(now_best_f1_ratio)
    setting = '{}_size{}_pdt{}_ctx{}_psz{}_bsz{}_test{}'.format(
        root_path.split('/')[-1],
        SIZE,
        PDT,
        CTX,
        PSZ,
        BSZ,
        TEST)
    file_path = os.path.join('output', setting, 'saved_middle_var.pkl')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path , 'wb') as pickle_file:
        pickle.dump(saved_middle_var, pickle_file)

    f1_dict_ratio = defaultdict(list)
    # 遍历列表，提取组合键和对应的 f1 值
    for entry in all_results_ratio:
        key = (entry['ratio'])
        f1_dict_ratio[key].append(entry['f1'])
    average_f1_ratio = {key: sum(f1_list) / len(f1_list) for key, f1_list in f1_dict_ratio.items()}
    print('ratio version')
    print("(ratio): average_f1")
    sorted_f1_ratio = sorted(average_f1_ratio.items(), key = lambda x:x[1])
    for f1 in sorted_f1_ratio:
        print(f1)
    # print("Best Global Ratio F1s: {:0.4f}, Best anomaly ratio : {:0.4f}".format(best_f1s, best_ratio))
    print("Best Indivisual Ratio F1s: {:0.4f}, Best anomaly ratios : {}".format(sum(all_best_f1s)/len(all_best_f1s), ["{:.4f}".format(item) for item in all_best_f1s_ratio]))

    # print("Ratio version over")

    # input_it = iter(test_data.input)
    # label_it = iter(test_data.label)
    # forecast_it = iter(forecasts)

    # inp = next(input_it)
    # label = next(label_it)
    # forecast = next(forecast_it)

    # plot_single(
    #     inp, 
    #     label, 
    #     forecast, 
    #     context_length=200,
    #     name="pred",
    #     show_label=True,
    # )
    # # plt.show()
    # plt.savefig("./output/moirai.pdf")