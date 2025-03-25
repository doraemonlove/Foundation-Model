import numpy as np

# 读取.npz文件
data = np.load('checkpoints/ALL_task_UniTS_pretrain_x128_UniTS_All_ftM_dm128_el3_Exp_0/Heat_M_p12_tt_predictions.npz')

# 获取文件中的数组
predictions = data['predictions']
ground_truth = data['ground_truth']

# 打印数组的形状
print("预测值(predictions)的维度:", predictions.shape)
print("真实值(ground_truth)的维度:", ground_truth.shape)

# 打印更详细的信息
print("\n详细信息:")
print(f"predictions: shape={predictions.shape}, dtype={predictions.dtype}")
print(f"ground_truth: shape={ground_truth.shape}, dtype={ground_truth.dtype}")

# 如果想看具体的数值范围
print("\n数值范围:")
print(f"predictions: min={predictions.min():.4f}, max={predictions.max():.4f}")
print(f"ground_truth: min={ground_truth.min():.4f}, max={ground_truth.max():.4f}")
