#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA 内核，用于执行量化操作
__global__ void quantize_kernel(float* x, const float* scale, const float* zero_point, float thd_neg, float thd_pos, int length) {
    // 使用共享内存加载 scale 和 zero_point
    __shared__ float s_scale;
    __shared__ float s_zero_point;

    if (threadIdx.x == 0) {
        s_scale = scale[0];
        s_zero_point = zero_point[0];
    }

    // 确保所有线程都在加载完成后继续执行
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        // 量化过程
        float q = s_zero_point + (x[idx] / s_scale);
        q = fminf(fmaxf(q, thd_neg), thd_pos);  // 相当于 torch.clamp
        x[idx] = roundf(q);                    // 相当于 round_pass
        x[idx] = s_scale * (x[idx] - s_zero_point);
    }
}

// CUDA 内核，用于执行量化操作
__global__ void quantize_all_kernel(const float* x, float* y, const float* scale, const float* zero_point, 
                                const float* thd_neg, const float* thd_pos, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        int idx = row * M + col;
        
        // 量化过程
        float q = zero_point[idx] + (x[idx] / scale[idx]);
        q = fminf(fmaxf(q, thd_neg[idx]), thd_pos[idx]);  // 相当于 torch.clamp
        y[idx] = roundf(q);                              // 相当于 round_pass
        y[idx] = scale[idx] * (y[idx] - zero_point[idx]);
    }
}


// 计算 scale 和 zero_point 的函数
std::tuple<torch::Tensor, torch::Tensor> calcScaleZeroPoint_cu(torch::Tensor min_val, torch::Tensor max_val, int num_bits = 8) {
    float qmin = 0.0f;
    float qmax = powf(2.0f, num_bits) - 1.0f;

    float scale_val;
    if (min_val.item<float>() != max_val.item<float>()) {
        scale_val = (max_val.item<float>() - min_val.item<float>()) / (qmax - qmin);
    } else {
        scale_val = 1.0f;
    }

    float zero_point_val = qmax - max_val.item<float>() / scale_val;

    if (zero_point_val < qmin) {
        zero_point_val = qmin;
    } else if (zero_point_val > qmax) {
        zero_point_val = qmax;
    }

    zero_point_val = roundf(zero_point_val);

    // 将标量值包装为张量并返回
    torch::Tensor scale = torch::tensor({scale_val}, torch::dtype(torch::kFloat32)).to(min_val.device());
    torch::Tensor zero_point = torch::tensor({zero_point_val}, torch::dtype(torch::kFloat32)).to(min_val.device());

    return std::make_tuple(scale, zero_point);
}

// 量化器的前向传播函数（不包含初始化部分）
torch::Tensor quantize_forward_cu(torch::Tensor x, torch::Tensor scale, torch::Tensor zero_point, float thd_neg, float thd_pos) {
    int length = x.size(0);

    // 调用量化内核
    quantize_kernel<<<(length + 1023) / 1024, 1024>>>(x.data_ptr<float>(), scale.data_ptr<float>(), zero_point.data_ptr<float>(), thd_neg, thd_pos, length);

    // CUDA设备同步，确保内核执行完成
    cudaDeviceSynchronize();

    return x;
}


torch::Tensor quantize_all_forward_cu(torch::Tensor x, torch::Tensor scale, torch::Tensor zero_point, torch::Tensor thd_neg, torch::Tensor thd_pos) {
    int N = x.size(0);
    int M = x.size(1);

    // 创建一个新的张量，用于存储量化后的结果
    torch::Tensor y = torch::empty_like(x);

    // 调用量化内核
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    quantize_all_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), scale.data_ptr<float>(), zero_point.data_ptr<float>(), 
        thd_neg.data_ptr<float>(), thd_pos.data_ptr<float>(), N, M);

    // CUDA设备同步，确保内核执行完成
    cudaDeviceSynchronize();

    return y;  // 返回新变量 y
}

