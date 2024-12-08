#include "utils.h"


torch::Tensor octree_fw(
    const torch::Tensor occodex,
    const torch::Tensor occodey,
    const torch::Tensor occodez,
    const torch::Tensor letrax,
    const torch::Tensor letray,
    const torch::Tensor letraz,
    const torch::Tensor indices
){
    CHECK_INPUT(occodex);
    CHECK_INPUT(occodey);
    CHECK_INPUT(occodez);
    CHECK_INPUT(letrax);
    CHECK_INPUT(letray);
    CHECK_INPUT(letraz);
    CHECK_INPUT(indices);

    return octree_fw_cu(occodex, occodey, occodez, letrax, letray, letraz, indices);
}


torch::Tensor octree_bw(
    const torch::Tensor dL_danchor_v,
    const torch::Tensor indices,
    const torch::Tensor ppoints
){
    CHECK_INPUT(dL_danchor_v);
    CHECK_INPUT(indices);
    CHECK_INPUT(ppoints);

    return octree_bw_cu(dL_danchor_v, indices, ppoints);
}


torch::Tensor feature_fw(
    const torch::Tensor& pfeatures,
    const torch::Tensor& indices, 
    const torch::Tensor& imp, 
    const std::string& oct_merge
){
    CHECK_INPUT(pfeatures);
    CHECK_INPUT(indices);
    CHECK_INPUT(imp);

    return feature_fw_cu(pfeatures, indices, imp, oct_merge);
}

torch::Tensor feature_bw(
    const torch::Tensor& grad_output, 
    const torch::Tensor& pfeatures, 
    const torch::Tensor& indices, 
    const torch::Tensor& imp, 
    const std::string& oct_merge
){
    CHECK_INPUT(grad_output)
    CHECK_INPUT(pfeatures);
    CHECK_INPUT(indices);
    CHECK_INPUT(imp);

    return feature_bw_cu(grad_output, pfeatures, indices, imp, oct_merge);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("octree_fw", &octree_fw);
    m.def("octree_bw", &octree_bw);
    m.def("feature_fw", &feature_fw);
    m.def("feature_bw", &feature_bw);
}
