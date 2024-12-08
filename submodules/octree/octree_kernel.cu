#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

// __global__ void mean_merge_kernel(const float* pfeatures, const int64_t* indices, float* output, int n, int feature_dim) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;  // 处理索引
//     int d = blockIdx.y * blockDim.y + threadIdx.y;  // 处理特征维度

//     if (i < n && d < feature_dim) {
//         int start = indices[i];
//         int end = indices[i + 1];
//         int length = end - start;

//         float sum = 0.0f;
//         for (int k = start; k < end; ++k) {
//             sum += pfeatures[k * feature_dim + d];
//         }
//         output[i * feature_dim + d] = sum / length;
//     }
// }

__global__ void mean_merge_kernel(const float* pfeatures, const int64_t* indices, float* output, int n, int feature_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int start = indices[i];
        int end = indices[i + 1];
        int length = end - start;
        for (int d = 0; d < feature_dim; ++d) {
            float sum = 0.0f;
            for (int k = start; k < end; ++k) {
                sum += pfeatures[k * feature_dim + d];
            }
            output[i * feature_dim + d] = sum / length;
        }
    }
}


__global__ void imp_merge_kernel(const float* pfeatures, const float* imp, const int64_t* indices, float* output, int n, int feature_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int start = indices[i];
        int end = indices[i + 1];
        float imp_sum = 0.0f;
        for (int k = start; k < end; ++k) {
            imp_sum += imp[k];
        }
        for (int d = 0; d < feature_dim; ++d) {
            float weighted_sum = 0.0f;
            for (int k = start; k < end; ++k) {
                weighted_sum += pfeatures[k * feature_dim + d] * imp[k];
            }
            output[i * feature_dim + d] = weighted_sum / imp_sum;
        }
    }
}

__global__ void mean_merge_backward_kernel(
    const float* grad_output, 
    const int64_t* indices, 
    float* grad_pfeatures, 
    int n, 
    int feature_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int start = indices[i];
        int end = indices[i + 1];
        int length = end - start;

        for (int k = start; k < end; ++k) {
            for (int d = 0; d < feature_dim; ++d) {
                grad_pfeatures[k * feature_dim + d] = grad_output[i * feature_dim + d] / length;
            }
        }
    }
}

__global__ void imp_merge_backward_kernel(
    const float* grad_output, 
    const float* imp, 
    const int64_t* indices, 
    float* grad_pfeatures, 
    int n, 
    int feature_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int start = indices[i];
        int end = indices[i + 1];
        float imp_sum = 0.0f;

        for (int k = start; k < end; ++k) {
            imp_sum += imp[k];
        }

        for (int k = start; k < end; ++k) {
            for (int d = 0; d < feature_dim; ++d) {
                float weighted_grad = grad_output[i * feature_dim + d] / imp_sum;
                grad_pfeatures[k * feature_dim + d] = weighted_grad * imp[k];
            }
        }
    }
}



__global__ void forward_kernel(int64_t* occodex, int64_t* occodey, int64_t* occodez, float* letra_x, float* letra_y, float* letra_z, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int64_t occ_x = occodex[idx];
        int64_t occ_y = occodey[idx];
        int64_t occ_z = occodez[idx];

        result[idx * 3 + 0] = letra_x[occ_x];
        result[idx * 3 + 1] = letra_y[occ_y];
        result[idx * 3 + 2] = letra_z[occ_z];
    }
}


__global__ void backward_kernel(float* grad_output, int64_t* indices, float* out_grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1) {
        int64_t start = indices[idx];
        int64_t end = indices[idx + 1];
        for (int i = start; i < end; ++i) {
            out_grad[i * 3 + 0] = grad_output[idx * 3 + 0];
            out_grad[i * 3 + 1] = grad_output[idx * 3 + 1];
            out_grad[i * 3 + 2] = grad_output[idx * 3 + 2];
        }
    }
}


torch::Tensor octree_fw_cu(torch::Tensor occodex, torch::Tensor occodey, torch::Tensor occodez, torch::Tensor letra_x, torch::Tensor letra_y, torch::Tensor letra_z, torch::Tensor indices) {
    auto result = torch::empty({occodex.size(0), 3}, torch::device(torch::kCUDA).dtype(torch::kFloat));

    int threads = 1024;
    int blocks = (occodex.size(0) + threads - 1) / threads;

    forward_kernel<<<blocks, threads>>>(
        occodex.data_ptr<int64_t>(), 
        occodey.data_ptr<int64_t>(), 
        occodez.data_ptr<int64_t>(), 
        letra_x.data_ptr<float>(), 
        letra_y.data_ptr<float>(), 
        letra_z.data_ptr<float>(), 
        result.data_ptr<float>(), 
        occodex.size(0)
    );

    return result;
}

torch::Tensor octree_bw_cu(torch::Tensor grad_output, torch::Tensor indices, torch::Tensor ppoints) {
    auto out_grad = torch::zeros({ppoints.size(0), 3}, torch::device(torch::kCUDA).dtype(torch::kFloat));

    int threads = 1024;
    int blocks = (ppoints.size(0) + threads - 1) / threads;

    backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(), 
        indices.data_ptr<int64_t>(), 
        out_grad.data_ptr<float>(), 
        indices.size(0)
    );

    return out_grad;
}

torch::Tensor feature_fw_cu(const torch::Tensor& pfeatures, const torch::Tensor& indices, const torch::Tensor& imp, const std::string& oct_merge) {
    int n = indices.size(0) - 1;
    int feature_dim = pfeatures.size(1);

    // Allocate output tensor
    auto output = torch::zeros({n, feature_dim}, pfeatures.options());

    int block_size = 256;  // 选择一个合适的块大小
    int grid_size = (n + block_size - 1) / block_size;  // 确保所有 n 个元素都被处理

    if (oct_merge == "mean") {
        mean_merge_kernel<<<grid_size, block_size>>>(
            pfeatures.data_ptr<float>(), 
            indices.data_ptr<int64_t>(), 
            output.data_ptr<float>(), 
            n, 
            feature_dim
        );
    } else if (oct_merge == "imp") {
        imp_merge_kernel<<<grid_size, block_size>>>(
            pfeatures.data_ptr<float>(), 
            imp.data_ptr<float>(), 
            indices.data_ptr<int64_t>(), 
            output.data_ptr<float>(), 
            n, 
            feature_dim
        );
    }

    // 同步和错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return torch::Tensor();
    }
    cudaDeviceSynchronize();
    return output;

}

torch::Tensor feature_bw_cu(
    const torch::Tensor& grad_output,    // Gradient of the loss w.r.t. the output
    const torch::Tensor& pfeatures,      // Input features from the forward pass
    const torch::Tensor& indices,        // Indices used in the forward pass
    const torch::Tensor& imp,            // Importance weights from the forward pass (if any)
    const std::string& oct_merge         // Merge strategy used in the forward pass
) {
    // Get dimensions
    int n = indices.size(0) - 1;
    int feature_dim = pfeatures.size(1);

    // Allocate tensors for gradients
    auto grad_pfeatures = torch::zeros({pfeatures.size(0), pfeatures.size(1)}, pfeatures.options());

    // Launch the appropriate kernel based on oct_merge
    if (oct_merge == "mean") {
        mean_merge_backward_kernel<<<(n + 255) / 256, 256>>>(
            grad_output.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            grad_pfeatures.data_ptr<float>(),
            n,
            feature_dim
        );
    } else if (oct_merge == "imp") {
        imp_merge_backward_kernel<<<(n + 255) / 256, 256>>>(
            grad_output.data_ptr<float>(),
            imp.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            grad_pfeatures.data_ptr<float>(),
            n,
            feature_dim
        );
    }

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA Error in backward kernel: " << cudaGetErrorString(err) << std::endl;
    //     return torch::Tensor();  // 或者采取其他适当的错误处理策略
    // }
    // cudaDeviceSynchronize();  // 确保内核执行完成

    return grad_pfeatures;
}