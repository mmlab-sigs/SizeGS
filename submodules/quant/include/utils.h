#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// 计算 scale 和 zero_point 的函数声明
std::tuple<torch::Tensor, torch::Tensor> calcScaleZeroPoint_cu(
    torch::Tensor min_val, 
    torch::Tensor max_val, 
    int num_bits = 8
);

// 量化器的前向传播函数声明
torch::Tensor quantize_forward_cu(
    torch::Tensor x, 
    torch::Tensor scale, 
    torch::Tensor zero_point, 
    float thd_neg, 
    float thd_pos
);

torch::Tensor quantize_all_forward_cu(
    torch::Tensor x, 
    torch::Tensor scale, 
    torch::Tensor zero_point, 
    torch::Tensor thd_neg, 
    torch::Tensor thd_pos
);