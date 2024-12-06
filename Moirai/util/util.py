from collections import defaultdict
import csv
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import argparse

warnings.filterwarnings('ignore')

from util.pot.pot import pot
from util.pot.spot import spot
import pandas as pd


def get_f1(score, label, thres):
    score = np.asarray(score)
    predict, actual = point_adjust(score, label, thres=thres)
    f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
    # print(predict)
    # print(len(predict))
    # print('actuall anomaly num: ', sum(actual), ', predicted anomaly sum: ', sum(predict))
    return (f1,precision,recall), (predict, actual)

def point_adjust(score, label, thres):
    if len(score) != len(label):
        raise ValueError("score len is %d and label len is %d\n" %(len(score), len(label)))
    
    score = np.asarray(score)
    label = np.asarray(label)
    predict = score > thres
    actual = label > 0.1
    anomaly_state = False
    
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
            
    return predict, actual

def calc_p2p(predict, actual):
    tp = np.sum(predict * actual)
    tn = np.sum((1-predict) * (1-actual))
    fp = np.sum(predict * (1-actual))
    fn = np.sum((1-predict) * actual)
    
    precision = tp / (tp + fp + 0.000001)
    recall = tp / (tp + fn + 0.000001)
    f1 = 2 * precision * recall / (precision + recall + 0.000001)
    return f1, precision, recall, tp, tn, fp, fn



def test_pot(score):
    score = np.asarray(score)
    print(score.shape)
    # 定义要测试的超参数
    # 对epsilon不敏感, 对num_candidates也不敏感(5/10)
    # 受risks影响最大(0.08最优)
    # 受init_level影响(init_level高于0.85时迅速,下降速度最快,低于0.85后波动不大)(0.78最优)
    # risks = [0.1, 0.08]
    risks = [0.25]
    # risks = np.arange(0.05, 0.95, 0.01)
    # risks = np.arange(0.60, 0.70, 0.001)
    # init_levels = np.arange(0.75, 0.85, 0.01)
    # init_levels = [0.5, 0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
    init_levels = [0.75]
    # num_candidates_list = [3, 5, 7, 10, 20]
    # epsilon_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
    # risks = [1e-2]
    # init_levels = [0.90]
    # init_levels = [0.8]
    num_candidates_list = [5]
    epsilon_list = [1e-8] 
    
    # init_levels = [np.arange(0.60, 0.70, 0.001)]
    # 遍历不同的超参数组合
    results = []
    for risk in risks:
        for init_level in init_levels:
            for num_candidates in num_candidates_list:
                for epsilon in epsilon_list:
                    z, t = pot(score, risk=risk, init_level=init_level, num_candidates=num_candidates, epsilon=epsilon)
                    # 保存超参数及对应的阈值结果
                    results.append({
                        'risk': risk,
                        'init_level': init_level,
                        'num_candidates': num_candidates,
                        'epsilon': epsilon,
                        'threshold': z,
                        'init_threshold': t
                    })
    return results