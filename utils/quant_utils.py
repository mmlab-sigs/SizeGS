import math
import time

import torch
import quant_cuda
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

import numpy as np

class Quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, thd_neg, thd_pos):
        return quant_cuda.quantize_forward(x, scale, zero_point, thd_neg, thd_pos)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

class Quant_all(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, thd_neg, thd_pos):
        return quant_cuda.quantize_all_forward(x, scale, zero_point, thd_neg, thd_pos)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def split_length(length, n):
    base_length = length / n
    floor_length = int(base_length)
    remainder = length - (floor_length * n)
    result = [floor_length + 1] * remainder + [floor_length] * (n - remainder)
    return result


def get_weight(segment_index, quantization_functions, C, start, end, total_segments, unit_length):
    # 获取子矩阵
    C_segment = C[unit_length*start:unit_length*end]
    total_diff = 0
    for i in range(C.shape[1]):
        # 使用对应的量化函数
        quantized_C = quantization_functions[i * total_segments + segment_index - 1](C_segment[:, i])
        # 计算误差
        diff = torch.sum((quantized_C - C_segment[:, i]) ** 2)
        total_diff += diff
    # 返回平均误差
    return total_diff / ((end - start) * unit_length)

def dp_split(C, quantization_functions, num_segments, total_length, unit_length):
    # 初始化缓存字典，用于保存子问题的结果
    cache = {}
    total_segments = num_segments
    initial_length = total_length
    
    # 计算下一步要填充的数量
    next_num = math.ceil(total_length / unit_length) * unit_length
    padded_length = next_num // unit_length  # 以1000为单位的长度
    segment_size = int(padded_length / num_segments)
    print(num_segments)
    print(segment_size)

    # 计算填充行数并填充
    padding_rows = next_num - C.shape[0]
    if padding_rows > 0:
        C = torch.cat([C, torch.zeros(padding_rows, C.shape[1], device=C.device)], dim=0)

    # 递归动态规划
    def recursive_dp(n, length):
        # 如果只剩一个分段，则直接计算
        if n == 1:
            return get_weight(n, quantization_functions, C, 0, length - 1, total_segments, unit_length), [length]
        
        # 检查缓存
        if (n, length) in cache:
            return cache[(n, length)]

        start = int(segment_size * 3 / 4)
        end = int(segment_size * 5 / 4)
        min_cost = float('inf')
        best_split_point = 0
        best_split_sequence = []

        # 尝试不同的分割点
        for i in range(start, end + 1):
            remaining_length = length - i
            if remaining_length < (n - 1) * start or remaining_length > (n - 1) * end:
                continue
            
            # 递归计算剩下的部分
            dp_cost, split_sequence = recursive_dp(n - 1, remaining_length)
            dp_cost += get_weight(n, quantization_functions, C, remaining_length, length, total_segments, unit_length)

            # 更新最小代价和最佳分割点
            if dp_cost < min_cost:
                min_cost = dp_cost
                best_split_point = i
                best_split_sequence = split_sequence[:]

        # 记录最优解并缓存
        best_split_sequence.append(best_split_point)
        cache[(n, length)] = (min_cost, best_split_sequence)
        return min_cost, best_split_sequence

    # 调用递归动态规划
    _, split_sequence = recursive_dp(num_segments, padded_length)
    
    # 还原分割点，转换回真实的长度
    split_sequence = [x * unit_length for x in split_sequence]
    split_sequence[-1] = split_sequence[-1] - unit_length + (initial_length % unit_length)
    
    return split_sequence


def seg_quant(x, split, qas, debug=False, need_all=False):
    if need_all:
        start = 0
        cnt = 0
        scales = torch.zeros_like(x)
        zero_points = torch.zeros_like(x)
        thd_negs = torch.zeros_like(x)
        thd_poss = torch.zeros_like(x)
        for length in split:
            scale, zero_point, thd_neg, thd_pos = qas[cnt](x[start:start+length], need_all)
            scales[start:start+length].fill_(scale.item())
            zero_points[start:start+length].fill_(zero_point.item())
            thd_negs[start:start+length].fill_(thd_neg)
            thd_poss[start:start+length].fill_(thd_pos)
            cnt += 1
            start += length
        return scales, zero_points, thd_negs, thd_pos
    else:
        segments = torch.split(x, split, dim=0)
        quantized_segments = [qas[i](segment) for i, segment in enumerate(segments)]
        return torch.cat(quantized_segments, dim=0)

def seg_quant_con(x, split, qas, debug=False):

    segments = torch.split(x, split, dim=0)
    quantized_segments = [qas(segment) for segment in segments]
    return torch.cat(quantized_segments, dim=0)

def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.
 
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    
    return q_x

def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x - zero_point)

def seg_quant_forward(x, split, qas):
    start = 0
    cnt = 0
    outs = []
    trans = []
    for length in split:
        i_scale = qas[cnt].scale
        i_zp = qas[cnt].zero_point
        i_bit = qas[cnt].bit
        outs.append(quantize_tensor(
            x[start:start+length], 
            scale=i_scale,
            zero_point=i_zp,
            num_bits=i_bit))
        # outs.append(Quant.apply(x[start:start+length], i_scale, i_zp, i_thd_neg, i_thd_pos)) 
        trans.extend([i_scale.item(), i_zp.item()])
        cnt += 1
        start += length
    return torch.concat(outs, dim=0), trans

def seg_quant_reverse(x, split, sz):
    cnt = 0 
    start = 0
    outs = []
    for length in split:
        i_scale = sz[cnt]
        i_zp = sz[cnt+1]
        outs.append(
            dequantize_tensor(
                x[start:start+length],
                scale=i_scale,
                zero_point=i_zp
            )
        )
        cnt+=2
        start += length
    return torch.concat(outs, axis=0)


class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
class ALSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta):
        # assert alpha > 0, "alpha={}".format(alpha)
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp
        w_q = Round.apply(torch.div((weight - beta), alpha).clamp(Qn, Qp))
        w_q = w_q * alpha + beta
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = (weight - beta) / alpha
        smaller = (q_w < Qn).float() #bool值转浮点值，1.0或者0.0
        bigger = (q_w > Qp).float() #bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller -bigger #得到位于量化区间的index
        grad_alpha = ((smaller * Qn + bigger * Qp + 
            between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        #在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight
        #返回的梯度要和forward的参数对应起来
        return grad_weight, grad_alpha,  None, None, None, grad_beta


class LSQPlusActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False,batch_init = 20):
        #activations 没有per-channel这个选项的
        super(LSQPlusActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** self.a_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (self.a_bits - 1)
            self.Qp = 2 ** (self.a_bits - 1) - 1
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        # self.beta = torch.nn.Parameter(torch.tensor([float(0)]))
        self.beta = torch.nn.Parameter(torch.tensor([float(-1e-9)]), requires_grad=True)
        self.init_state = 0

    # 量化/反量化
    def forward(self, activation):
        #V1
        # print(self.a_bits, self.batch_init)
        if self.a_bits == 32:
            q_a = activation
        elif self.a_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits != 1
        else:
            if self.init_state==0:
                self.g = 1.0/math.sqrt(activation.numel() * self.Qp)
                self.init_state += 1
            q_a = ALSQPlus.apply(activation, self.s, self.g, self.Qn, self.Qp, self.beta)
            # print(self.s, self.beta)
        return q_a

def grad_scale(x, scale):
    y = x
    y_grad = x * scale 
    return (y - y_grad).detach() + y_grad 

def round_pass(x):
    y = x.round()
    y_grad = x 
    return (y - y_grad).detach() + y_grad

class Quantizer(nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError


class IdentityQuan(Quantizer):
    def __init__(self, bit=None, *args, **kwargs):
        super().__init__(bit)
        assert bit is None, 'The bit-width of identity quantizer must be None'

    def forward(self, x):
        return x


class LsqQuan(Quantizer):
    def __init__(self, bit, init_yet, all_positive=True, symmetric=False, per_channel=False):
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

        self.per_channel = per_channel
        self.s = nn.Parameter(torch.ones(1))
        self.init_yet = init_yet
    
    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
        self.init_yet = True
        # print('quant_utils.py Line 62:', self.s)
    
    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x


class LearnableScaledRoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x, scale)
        scaled_x = x * scale
        return torch.round(scaled_x) / scale
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        # 计算对x和scale的梯度
        grad_x = grad_output.clone() * scale
        grad_scale = (grad_output.clone() * x).sum()  # 根据需求调整梯度计算
        return grad_x, grad_scale

# 包装为可学习模块
class LearnableQuantizer(nn.Module):
    def __init__(self, init_scale=2**12):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale))
    
    def forward(self, x):
        return LearnableScaledRoundSTE.apply(x, self.scale)
    

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


class VanillaQuan(Quantizer):
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
        
        scale_cu, zero_point_cu = quant_cuda.calcScaleZeroPoint(self.min_val, self.max_val, self.bit)
        self.scale = scale_cu
        self.zero_point = zero_point_cu
    
    def forward(self, x, need_all=False):
        self.update(x)
        if need_all:
            return self.scale, self.zero_point, self.thd_neg, self.thd_pos
        else:
            q_x = Quant.apply(x.contiguous(), self.scale, self.zero_point, self.thd_neg, self.thd_pos)
            return q_x