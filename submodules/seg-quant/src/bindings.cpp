#include <torch/extension.h>

#include "seg_quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_pass", &quant_pass, "quantization simulation");
    m.def("quant_forward", &quant_forward, "apply quantization");
    m.def("quant_reverse", &quant_reverse, "apply dequantization");
    m.def("cal_diff", &cal_diff, "calculate difference");
}