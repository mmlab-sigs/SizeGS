#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void quant_kernel(
    const int C,
    const int N,
    const int B,
    const float * __restrict__ features, 
    const int * __restrict__ splits, 
    const int * __restrict__ power_qbits, 
    float* min_vals,
    float* max_vals,
    float* out
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < B * C) {
        int c = idx / B;
        int start_idx = 0;
        if (idx > 0) {
            start_idx = splits[idx - 1];
        }
        int end_idx = splits[idx];
        float min_val = min_vals[idx];
        // printf("min_val %f\n", min_val);
        float max_val = max_vals[idx];
        int qmax = power_qbits[c];
        int quant_length = end_idx - start_idx;
        float tmp = 0;
        const float* features_ptr = features + c * N + start_idx;
        float* out_ptr = out + c * N + start_idx;

        for (int i = 0; i < quant_length; i++) {
            tmp = features_ptr[i];
            if (tmp < min_val) {
                min_val = tmp;
            }
            if (tmp > max_val) {
                max_val = tmp;
            }
        }

        if (min_val > 0) {
            min_val = 0;
        }
        if (max_val < 0) {
            max_val = 0;
        }
        
        // float scale = 1.0f;
        float scale = 1.0f;
        if (min_val != max_val) {
            scale = (max_val - min_val) / (qmax - 0.);
            if (scale < 1e-12f) {
                scale = 1e-12f;  // 或根据需求调整
            }
        } 
        
        float zero_point = qmax - max_val / scale;
        if (zero_point < 0.0f) {
            zero_point = 0.0f;
        } else if (zero_point > qmax) {
            zero_point = qmax;
        }
        zero_point = roundf(zero_point);

        for (int i = 0; i < quant_length; i++) {
            float q = zero_point + (features_ptr[i] / scale);
            q = roundf(fminf(fmaxf(q, 0), qmax));
            out_ptr[i] = scale * (q - zero_point); // save value to out features
        }

        // save new value to max_val and min_val
        min_vals[idx] = min_val;
        max_vals[idx] = max_val;
    }
}

__global__ void quant_forward_kernel(
    const int C,
    const int N,
    const int B,
    const float * __restrict__ features, 
    const int * __restrict__ splits, 
    const int * __restrict__ power_qbits, 
    float* min_vals,
    float* max_vals,
    float* out,
    float* scales,
    float* zero_points
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < B * C) {
        int c = idx / B;
        int start_idx = 0;
        if (idx > 0) {
            start_idx = splits[idx - 1];
        }
        int end_idx = splits[idx];
        float min_val = min_vals[idx];
        // printf("min_val %f\n", min_val);
        float max_val = max_vals[idx];
        float qmax = power_qbits[c];
        
        int quant_length = end_idx - start_idx;
        float tmp = 0;
        const float* features_ptr = features + c * N + start_idx;
        float* out_ptr = out + c * N + start_idx;
        
        float scale = 1.0f;
        if (min_val != max_val) {
            scale = (max_val - min_val) / (qmax - 0.);
        }
        
        float zero_point = qmax - max_val / scale;
        if (zero_point < 0.0f) {
            zero_point = 0.0f;
        } else if (zero_point > qmax) {
            zero_point = qmax;
        }
        zero_point = roundf(zero_point);

        for (int i = 0; i < quant_length; i++) {
            float q = zero_point + (features_ptr[i] / scale);
            out_ptr[i] = roundf(fminf(fmaxf(q, 0), qmax)); // save value to out features
        }

        // save new value to max_val and min_val, do not update here.
        // min_vals[idx] = min_val;
        // max_vals[idx] = max_val;
        scales[idx] = scale;
        zero_points[idx] = scale;
    }
}

__global__ void quant_reverse_kernel(
    const int C,
    const int N,
    const int B,
    const float * __restrict__ features, 
    const int * __restrict__ splits, 
    float* scales,
    float* zero_points,
    float* out
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < B * C) {
        int c = idx / B;
        int start_idx = 0;
        if (idx > 0) {
            start_idx = splits[idx - 1];
        }
        int end_idx = splits[idx];
        float scale = scales[idx];
        float zero_point = zero_points[idx];
        
        int quant_length = end_idx - start_idx;
        const float* features_ptr = features + c * N + start_idx;
        float* out_ptr = out + c * N + start_idx;
        
        for (int i = 0; i < quant_length; i++) {
            out_ptr[i] = scale * (features_ptr[i] - zero_point);
        }

        // save new value to max_val and min_val, do not update here.
        // min_vals[idx] = min_val;
        // max_vals[idx] = max_val;
    }
}

void quant_pass(
    const torch::Tensor& features,
    const torch::Tensor& splits,
    const torch::Tensor& power_qbits,
    torch::Tensor& min_vals,
    torch::Tensor& max_vals,
    torch::Tensor& out
) {
    const int C = features.size(0);
    const int N = features.size(1);
    const int B = splits.size(1);

    quant_kernel <<<ceil(C * B / 256.0), 256>>>(
        C, 
        N,
        B,
        features.data_ptr<float>(),
        splits.data_ptr<int>(),
        power_qbits.data_ptr<int>(),
        min_vals.data_ptr<float>(),
        max_vals.data_ptr<float>(),
        out.data_ptr<float>()
    );
}

void quant_forward(
    const torch::Tensor& features,
    const torch::Tensor& splits,
    const torch::Tensor& power_qbits,
    const torch::Tensor& min_vals,
    const torch::Tensor& max_vals,
    torch::Tensor& out,
    torch::Tensor& scales,
    torch::Tensor& zero_points
) {
    const int C = features.size(0);
    const int N = features.size(1);
    const int B = splits.size(1);

    quant_forward_kernel <<<ceil(C * B / 256.0), 256>>>(
        C, 
        N,
        B,
        features.data_ptr<float>(),
        splits.data_ptr<int>(),
        power_qbits.data_ptr<int>(),
        min_vals.data_ptr<float>(),
        max_vals.data_ptr<float>(),
        out.data_ptr<float>(),
        scales.data_ptr<float>(),
        zero_points.data_ptr<float>()
    );
}

void quant_reverse(
    const torch::Tensor& features,
    const torch::Tensor& splits,
    const torch::Tensor& scales,
    const torch::Tensor& zero_points,
    torch::Tensor& out
) {
    const int C = features.size(0);
    const int N = features.size(1);
    const int B = splits.size(1);
    // const int num_blocks = (C * B + 255) / 256;
    quant_reverse_kernel <<<ceil(C * B / 256.0), 256>>>(
        C, 
        N,
        B,
        features.data_ptr<float>(),
        splits.data_ptr<int>(),
        scales.data_ptr<float>(),
        zero_points.data_ptr<float>(),
        out.data_ptr<float>()
    );
}


__global__ void cal_diff_kernel(
    const int C,
    const int N,
    const int B,
    const float * __restrict__ features, 
    const int * __restrict__ cum_splits, 
    const int * __restrict__ power_qbits, 
    float* min_vals,
    float* max_vals,
    float* out_diff
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < B * C) {
        int c = idx / B;
        int start_idx = 0;
        if (idx > 0) {
            start_idx = cum_splits[idx - 1];
        }
        int end_idx = cum_splits[idx];
        float min_val = min_vals[idx];
        // printf("min_val %f\n", min_val);
        float max_val = max_vals[idx];
        int qmax = power_qbits[idx];
        int quant_length = end_idx - start_idx;
        float tmp = 0;
        const float* features_ptr = features + c * N + start_idx;

        for (int i = 0; i < quant_length; i++) {
            tmp = features_ptr[i];
            if (tmp < min_val) {
                min_val = tmp;
            }
            if (tmp > max_val) {
                max_val = tmp;
            }
        }

        if (min_val > 0) {
            min_val = 0;
        }
        if (max_val < 0) {
            max_val = 0;
        }
        
        // float scale = 1.0f;
        float scale = 1.0f;
        if (min_val != max_val) {
            scale = (max_val - min_val) / (qmax - 0.);
            if (scale < 1e-12f) {
                scale = 1e-12f;  // 或根据需求调整
            }
        } 
        
        float zero_point = qmax - max_val / scale;
        if (zero_point < 0.0f) {
            zero_point = 0.0f;
        } else if (zero_point > qmax) {
            zero_point = qmax;
        }
        zero_point = roundf(zero_point);
        
        float diff = 0.0f;
        for (int i = 0; i < quant_length; i++) {
            float q = zero_point + (features_ptr[i] / scale);
            q = roundf(fminf(fmaxf(q, 0), qmax));
            diff = diff + abs(scale * (q - zero_point) - features_ptr[i]);
        }

        // save new value to max_val and min_val
        min_vals[idx] = min_val;
        max_vals[idx] = max_val;
        out_diff[idx] = diff;
    }
}

void cal_diff(
    const torch::Tensor& features,
    const torch::Tensor& cum_splits,
    const torch::Tensor& power_qbits,
    torch::Tensor& min_vals,
    torch::Tensor& max_vals,
    torch::Tensor& out_diff
) {
    const int C = features.size(0);
    const int N = features.size(1);
    const int B = cum_splits.size(1);

    cal_diff_kernel <<<ceil(C * B / 256.0), 256>>>(
        C, 
        N,
        B,
        features.data_ptr<float>(),
        cum_splits.data_ptr<int>(),
        power_qbits.data_ptr<int>(),
        min_vals.data_ptr<float>(),
        max_vals.data_ptr<float>(),
        out_diff.data_ptr<float>()
    );
}


// __global__ void max_reduction_kernel(int* features, int* cum_nsplits, int* qbits, ) {
//     extern __shared__ int shared_mem[];
    
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     int tid = threadIdx.x;

//     // Load elements into shared memory
//     if (idx < n) {
//         shared_mem[tid] = arr[idx];
//     } else {
//         shared_mem[tid] = INT_MIN; // Use a small number if out of bounds
//     }
//     __syncthreads();

//     // Perform reduction in shared memory
//     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             shared_mem[tid] = max(shared_mem[tid], shared_mem[tid + s]);
//         }
//         __syncthreads();
//     }

//     // Store the result of each block in global memory
//     if (tid == 0) {
//         result[blockIdx.x] = shared_mem[0];
//     }
// }