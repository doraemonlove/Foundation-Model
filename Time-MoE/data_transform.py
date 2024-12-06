import pandas as pd
import numpy as np
import json
import os

file_dir = 'dataset/electricity'
# 读取 CSV 文件
df = pd.read_csv(os.path.join(file_dir, 'electricity.csv'))
df = df[['OT']]

# 添加数据比例限制
ratio = 0.1  # 使用前80%的数据
data_length = len(df)
truncated_length = int(data_length * ratio)
df = df.iloc[:truncated_length]

data = df.values.astype(np.float32)

# 设置窗口参数
window_size = 512
step = 16  # 添加窗口移动步长
feature_dim = data.shape[1]  # 特征维度

# 计算可以形成多少个窗口（考虑step）
num_sequences = (len(data) - window_size) // step + 1

# 准备元数据
meta = {
    "num_sequences": num_sequences,  # 窗口数量
    "dtype": "float32",
    "files": {
        "data-1-of-1.bin": len(data) * feature_dim  # 总数据点数
    },
    "scales": []
}

# 生成每个窗口的offset和length信息
for i in range(num_sequences):
    meta["scales"].append({
        "offset": i * step * feature_dim,  # 使用step来计算偏移量
        "length": window_size * feature_dim  # 窗口中的数据点总数
    })

# 保存元数据
with open(os.path.join(file_dir, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

# 保存数据为二进制文件
data.tofile(os.path.join(file_dir, 'data-1-of-1.bin'))
