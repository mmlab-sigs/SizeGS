import torch 
import torch.nn as nn
from utils.quant_utils import cal_diff_wrapper 
import numpy as np
npc = torch.tensor(np.load('exp_data/npc.npy'), device="cuda")
splits = torch.tensor(np.load('exp_data/splits.npy'), device="cuda")
n_channel = 73
n_block = 60
out = cal_diff_wrapper(
    channel_num = n_channel,
    block_num = n_block,
    inputs = npc,
    splits = splits,
    qbits = torch.ones([n_channel, n_block], device="cuda")*1,
)

torch.cuda.synchronize()

print(out)