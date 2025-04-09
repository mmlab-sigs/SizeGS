# pragma once 

#include <stdint.h>
#include <torch/torch.h>


void quant_pass(
    const torch::Tensor& features,
    const torch::Tensor& splits,
    const torch::Tensor& power_qbits,
    torch::Tensor& min_vals,
    torch::Tensor& max_vals,
    torch::Tensor& out
); 

void quant_forward(
    const torch::Tensor& features,
    const torch::Tensor& splits,
    const torch::Tensor& power_qbits,
    const torch::Tensor& min_vals,
    const torch::Tensor& max_vals,
    torch::Tensor& out,
    torch::Tensor& scales,
    torch::Tensor& zero_points
);

void quant_reverse(
    const torch::Tensor& features,
    const torch::Tensor& splits,
    const torch::Tensor& scales,
    const torch::Tensor& zero_points,
    torch::Tensor& out
);

void cal_diff(
    const torch::Tensor& features,
    const torch::Tensor& cum_splits,
    const torch::Tensor& power_qbits,
    torch::Tensor& min_vals,
    torch::Tensor& max_vals,
    torch::Tensor& out_diff
);

void quant_forward_wo_minmax(
    const torch::Tensor& features,
    const torch::Tensor& splits,
    const torch::Tensor& power_qbits,
    torch::Tensor& out,
    torch::Tensor& scales,
    torch::Tensor& zero_points,
    torch::Tensor& max_vals,
    torch::Tensor& min_vals
);


void cal_diff_infinity(
    const torch::Tensor& features,
    const torch::Tensor& cum_splits,
    const torch::Tensor& power_qbits,
    torch::Tensor& min_vals,
    torch::Tensor& max_vals,
    torch::Tensor& out_diff
);

void cal_diff_square(
    const torch::Tensor& features,
    const torch::Tensor& cum_splits,
    const torch::Tensor& power_qbits,
    torch::Tensor& min_vals,
    torch::Tensor& max_vals,
    torch::Tensor& out_diff
);