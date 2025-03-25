import numpy as np 
import torch
fq = np.load('/home/szxie/SizeGS/exp_data/final_qbits.npy')

f = open('exp_data/fq.txt', 'w')
for i in range(fq.shape[0]):
    for j in range(fq.shape[1]):
        f.write(f'{fq[i, j]} ')
    f.write("\n")
f.close()
print(fq)


# # for i in range(A.shape[0]):
# #     print(A[i])

# qf8 = np.load('/home/szxie/SizeGS/exp_data/qf_8.npy')

# cc = np.load('/home/szxie/SizeGS/exp_data/channel_check.npy')

# # print(qf8.shape)

# print(cc)

# dc ac_min ac_max ac_std ac_mean
# 'bicycle', 
# f = open('exp_data/channel_sq_sum_after.txt', 'w')
# # 'bicycle',
# for scene in ['bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump', 'flowers', 'treehill']:
#     brf = torch.tensor(np.load(f'exp_data/{scene}_before_raht.npy')).cuda()
    
#     arf = torch.tensor(np.load(f'exp_data/{scene}_after_raht.npy')).cuda()
    
#     arf_dc = arf[0].reshape(-1).tolist()
#     arf_ac = arf[1:]
#     f.write(scene)
#     f.write('\n')
#     # for ele in arf_dc.reshape(-1).tolist():
#     #     f.write(f"{ele} ")
#     # f.write("\n")
    
#     arf_min = torch.min(arf_ac, dim=0).values.reshape(-1).tolist()
#     arf_max = torch.max(arf_ac, dim=0).values.reshape(-1).tolist()
#     arf_mean = torch.mean(arf_ac, dim=0).reshape(-1).tolist()
#     arf_std = torch.std(arf_ac, dim=0).reshape(-1).tolist()
#     # print(arf_std)
#     for i in range(len(arf_std)):
#         f.write(f'{i}: dc-{arf_dc[i]} ac_min-{arf_min[i]} ac_max-{arf_max[i]} ac_std-{arf_std[i]} ac_mean-{arf_mean[i]}\n') # ac_max ac_std ac_mean
#     # srf = torch.sum(torch.square(rf), dim=0)
#     # f.write(scene)
#     # f.write('\n')
#     # print(srf.shape)
#     # lsrf = srf.cpu().tolist()
#     # for i in range(srf.shape[0]):
#     #     f.write(f'{i}, {lsrf[i]}\n')
#     break

# f.close()