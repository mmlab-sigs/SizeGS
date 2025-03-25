from utils.quant_utils import pure_quant_wo_minmax, split_length
from utils.compression import gpcc_decode, read_ply_geo_bin
import torch 
import os
import numpy as np 

bin_dir = '../zip'
tmp_dir = 'exp_data/tmp'
os.makedirs(tmp_dir, exist_ok=True)

ply_path = os.path.join(tmp_dir, 'voxelized_means.ply')
means_bin_path = os.path.join(bin_dir, 'compressed.bin')
gpcc_decode(bin_path=means_bin_path, recon_path=ply_path)
voxelized_means = read_ply_geo_bin(ply_path=ply_path).astype(np.float32)
n_point = voxelized_means.shape[0]

attris = np.load(os.path.join(bin_dir, 'fe.npz'))
trans = attris['t']
depth = int(trans[0])
n_block = int(trans[1])
n_channel = 73
enable_drop = True
if enable_drop: 
    n_c = n_channel - 3
else:
    n_c = n_channel

split_per_channel = split_length(n_point-1, n_block)
split_per_channel = torch.tensor(split_per_channel, device="cuda")
splits = split_per_channel.reshape(1, -1).repeat(n_c, 1)

oct_path = os.path.join(bin_dir, 'compressed.bin')
oct_bits = os.path.getsize(oct_path)
mlp_paths = [
    os.path.join(bin_dir, x) for x in ['color_mlp.pt', 'cov_mlp.pt', 'opacity_mlp.pt']
]
oct_param = attris['o']

def cal_entropy(rf, splits: torch.Tensor, input_qbits: torch.Tensor):
    # input_qbits = torch.tensor(input_qbits, dtype=int, device="cuda")
    power_qbits = torch.pow(2, input_qbits).int() - 1
    outputs, scales, zeros = pure_quant_wo_minmax(rf, splits, power_qbits)
    sz = torch.concat([scales.reshape(-1), zeros.reshape(-1)], dim = 0)
    _, counts = torch.unique(outputs, return_counts=True)
    n_features = outputs.shape[0] * outputs.shape[1]
    freq_prob = counts.float() / n_features 
    bit_per_fea = - torch.sum(freq_prob * torch.log2(freq_prob))
    qbsum = bit_per_fea * n_features
    return outputs, qbsum, sz

def save_compressed(
    save_dir, 
    ff: torch.Tensor, 
    qf: torch.Tensor, 
    sz: list, 
    depth: int, 
    n_channel: int,
    n_block: int, 
    channel_check: torch.Tensor, 
    split_one_channel: torch.Tensor,
    oct_param,
    enable_drop=True,
):
    save_bin_dir = os.path.join(save_dir, 'bin')
    trans = [depth, n_block]
    trans.extend(sz)
    qf_8 = []
    qf_16 = []
    sum_l = 0
    
    if enable_drop:
        n_c = n_channel-3
    else:
        n_c = n_channel
    cum_split = torch.sum(split_one_channel).int()
    for c in range(n_c):
        for b in range(n_block):
            if b == 0:
                si = 0
            else:
                si = int(cum_split[b-1].item())
            ei = int(cum_split[b].item())
            # print(si, ei)
            if channel_check[c, b]:
                sum_l += int(qf[c, si:ei].shape[-1])
                # print(qf[c, si:ei])
                qf_8.append(qf[c, si:ei].clone())
            else:
                sum_l += int(qf[c, si:ei].shape[-1])
                qf_16.append(qf[c, si:ei].clone())
    
    np.savez_compressed(
        os.path.join(save_bin_dir, 'fe'), 
        f=ff.cpu().numpy(), 
        i_8=qf_8, 
        i_16=qf_16, 
        t=trans, 
        h=channel_check, 
        o=oct_param)#这里需要改一下
    os.system(f'cp {oct_path} {save_bin_dir}/compressed.bin')
    for mlp_path in mlp_paths:
        mlp_name = mlp_path.split('/')[-1]
        os.system(f'cp {mlp_path} {save_bin_dir}/{mlp_name}')
    bin_zip_path = os.path.join(save_dir, 'bins.zip')
    os.system(f'zip -j {bin_zip_path} {save_bin_dir}/*')
    zip_bytes = os.path.getsize(bin_zip_path)
    
    return zip_bytes

def cal_est_size(qbits: list, splits: list, n_channel: int, n_block: int, oct_bits: int):
    trans_size = n_channel * n_block * 2 * 32 + 2 * 32 # scale and zero points, n_block, depth
    channel_check = n_channel * n_block * 1 # maybe 1 bit for True/False
    oct_param = 6 * 32 # min max in x, y, z 
    dc_size = n_channel * 32 # DC coefficient
    # oct_size = n_point * 64 # octree size 
    mlp_size = int(50 * 1024 * 8 * 0.75) # MLPs
    remain_bits = mlp_size + trans_size + channel_check + oct_param + dc_size + oct_bits
    for i in range(n_channel):
        for j in range(n_block):
            qbsz += qbits[i, j] * splits[i, j]
    return remain_bits + qbsz
    
for scene in ["bicycle"]:
    rf = torch.tensor(np.load(f'exp_data/{scene}_rf.npy'), device='cuda')
    