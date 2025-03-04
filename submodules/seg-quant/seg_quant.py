import torch 
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable 
from torch.cuda.amp import custom_bwd, custom_fwd 
import time 
import numpy as np

from backend import * 

def round_pass(x):
    y = x.round()
    y_grad = x 
    return (y - y_grad).detach() + y_grad

class SegQuant(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx, 
        inputs: torch.Tensor, 
        splits: torch.Tensor, 
        qbits: torch.Tensor, 
        min_vals: torch.Tensor, 
        max_vals: torch.Tensor
    ):
        inputs = inputs.contiguous() 
        splits = splits.contiguous()
        min_vals = min_vals.contiguous()
        max_vals = max_vals.contiguous()
        qbits = torch.pow(2, qbits).int().contiguous() - 1
        outputs = torch.empty_like(inputs).contiguous() 
        _backend.quant_pass(
            inputs, 
            splits,
            qbits,
            min_vals,
            max_vals,
            outputs
        )

        return outputs
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        return grad, None, None, None, None


def cal_diff_wrapper(
    inputs: torch.Tensor, 
    splits: torch.Tensor, 
    qbits: torch.Tensor
):
    inputs = inputs.transpose(0, 1).contiguous() 
    splits = splits.contiguous()
    min_vals = torch.zeros([channel_num, block_num]).contiguous().cuda()
    max_vals = torch.zeros([channel_num, block_num]).contiguous().cuda()
    qbits = torch.pow(2, qbits).int().contiguous() - 1
    outputs = torch.zeros([channel_num, block_num]).contiguous().cuda()
    _backend.cal_diff(
        inputs, 
        splits,
        qbits,
        min_vals,
        max_vals,
        outputs
    )
    
    return outputs

class BatchQuant(nn.Module):
    def __init__(self, channel_num: int, block_num: int):
        super().__init__()
        self.channel_num = channel_num
        self.block_num = block_num
        min_vals = torch.zeros([channel_num, block_num]).requires_grad_(False)
        max_vals = torch.zeros([channel_num, block_num]).requires_grad_(False)
        
        self.register_buffer('min_vals', min_vals) 
        self.register_buffer('max_vals', max_vals)

    def forward(self, inputs: torch.Tensor, splits: torch.Tensor, qbits: torch.Tensor):
        inputs = inputs.transpose(0, 1)
        # assert inputs.shape[0] == self.channel_num 
        outputs = SegQuant.apply(inputs, splits, qbits, self.min_vals, self.max_vals)
        return outputs.transpose(0, 1)
    
    
def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    if min_val != max_val:
        scale = (max_val - min_val) / (qmax - qmin)
    else:
        scale = torch.tensor([1], dtype=torch.float32).to(min_val.device)

    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax:
        # zero_point = qmax
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)
    
    zero_point.round_()

    return scale, zero_point

class VanillaQuan(nn.Module):
    def __init__(self, bit, all_positive=True, symmetric=False):
        super().__init__(bit)
        
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.bit = bit
        scale = None
        zero_point = None
        min_val = torch.tensor([], requires_grad=False)
        max_val = torch.tensor([], requires_grad=False)
        
        self.register_buffer('scale', scale)
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('min_val', min_val) 
        self.register_buffer('max_val', max_val)
        
    def update(self, x):
        
        if self.max_val.nelement() == 0 or self.max_val.data < x.max().data:
            self.max_val.data = x.max().data
        self.max_val.clamp_(min=0)
        
        if self.min_val.nelement() == 0 or self.min_val.data > x.min().data:
            self.min_val.data = x.min().data 
        self.min_val.clamp_(max=0)    
        
        scale_cu, zero_point_cu = calcScaleZeroPoint(self.min_val, self.max_val, self.bit)
        self.scale = scale_cu
        self.zero_point = zero_point_cu
    
    def forward(self, x):
        self.update(x)
        x = self.zero_point + (x / self.scale)
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = self.scale * (x - self.zero_point)

if __name__ == '__main__':
    
    x = np.load('/home/szxie/SizeGS/exp_data/C.npy')
    x = torch.tensor(x)[1:, 1:3].cuda()
    channel_num = 2
    block_num = 60
    N = x.shape[0] - 1
    
    br = time.time()
    bquant = BatchQuant(channel_num, block_num).cuda()
    # print('init batchquant', time.time() - br)
    # x = torch.randn([N, channel_num]).float().cuda()
    print('x', x)
    # qbits = torch.randint(low=1, high=16, size=[channel_num]).cuda()
    qbits = torch.tensor([10]*channel_num).cuda()
    print(qbits)
     
    base_length = N / block_num
    floor_length = int(base_length)
    remainder = N - (floor_length * block_num)
    splits = [floor_length + 1] * remainder + [floor_length] * (block_num - remainder)
    splits = torch.tensor(splits).int().cuda()
    splits = torch.cumsum(splits, dim=0).int().reshape(1, -1).repeat(channel_num, 1)
    print(splits[0])
    # print(splits[1])
    print(splits.shape)
    # print(splits[-1], '==', N)
    
    br = time.time()
    # o = bquant(x, splits, qbits) 
    o = cal_diff_wrapper(x, splits, qbits)
    torch.cuda.synchronize()
    print('infer time', time.time() - br)
    print(o.shape)
    print('o', o)
    # print('distance', torch.square(o - x))
    
