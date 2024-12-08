#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor octree_fw_cu(
    const torch::Tensor occodex,
    const torch::Tensor occodey,
    const torch::Tensor occodez,
    const torch::Tensor letrax,
    const torch::Tensor letray,
    const torch::Tensor letraz,
    const torch::Tensor indices
);


torch::Tensor octree_bw_cu(
    const torch::Tensor dL_danchor_v,
    const torch::Tensor indices,
    const torch::Tensor ppoints
);


torch::Tensor feature_fw_cu(
    const torch::Tensor& pfeatures,
    const torch::Tensor& indices, 
    const torch::Tensor& imp, 
    const std::string& oct_merge
);


torch::Tensor feature_bw_cu(
    const torch::Tensor& grad_output, 
    const torch::Tensor& pfeatures,
    const torch::Tensor& indices, 
    const torch::Tensor& imp, 
    const std::string& oct_merge
);