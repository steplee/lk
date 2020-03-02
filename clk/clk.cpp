#include "clk.h"
#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

RegistrationResult::RegistrationResult() {
}

RegistrationResult registerAffine_call_cuda(torch::Tensor& t);
RegistrationResult registerAffine(torch::Tensor& t) {
  return registerAffine_call_cuda(t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<RegistrationResult>(m, "RegistrationResult");

    m.def("registerAffine", &registerAffine, "register (affine)");
}
