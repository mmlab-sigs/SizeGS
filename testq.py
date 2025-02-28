import torch 
import torch.nn as nn


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


class RAHT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, voxel_xyz, features):
        
        pass