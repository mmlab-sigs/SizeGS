import numpy as np

# 文件路径
file1 = "/home/szxie/sizegs_private/outputs/360_v2/bicycle/delta_tau/2025-06-14_19:49:54/ingredients/final_qbits.npy"
file2 = "/home/szxie/sizegs_private/outputs/360_v2/bicycle/rebuttal_auto_tau/2025-06-19_22:57:16/ingredients/final_qbits.npy"

# 加载数据
data1 = np.load(file1)
data2 = np.load(file2)

# 计算均值
mean1 = np.mean(data1)
mean2 = np.mean(data2)

print(f"Mean of {file1} : {mean1}")
print(f"Mean of {file2} : {mean2}")