#include "utils.h"

// 计算 scale 和 zero_point 的函数声明
std::tuple<torch::Tensor, torch::Tensor> calcScaleZeroPoint(
    torch::Tensor min_val, 
    torch::Tensor max_val, 
    int num_bits = 8
){
    CHECK_INPUT(min_val);
    CHECK_INPUT(max_val);

    return calcScaleZeroPoint_cu(min_val, max_val, num_bits);
}

// 量化器的前向传播函数声明
torch::Tensor quantize_forward(
    torch::Tensor x, 
    torch::Tensor scale, 
    torch::Tensor zero_point, 
    float thd_neg, 
    float thd_pos
){
    CHECK_INPUT(x);
    CHECK_INPUT(scale);
    CHECK_INPUT(zero_point);

    return quantize_forward_cu(x, scale, zero_point, thd_neg, thd_pos);
}

torch::Tensor quantize_all_forward(
    torch::Tensor x, 
    torch::Tensor scale, 
    torch::Tensor zero_point, 
    torch::Tensor thd_neg, 
    torch::Tensor thd_pos
){
    CHECK_INPUT(x);
    CHECK_INPUT(scale);
    CHECK_INPUT(zero_point);
    CHECK_INPUT(thd_neg);
    CHECK_INPUT(thd_pos);

    return quantize_all_forward_cu(x, scale, zero_point, thd_neg, thd_pos);
}



// 定义扩展模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_forward", &quantize_forward);
    m.def("calcScaleZeroPoint", &calcScaleZeroPoint);
    m.def("quantize_all_forward", &quantize_all_forward);
}