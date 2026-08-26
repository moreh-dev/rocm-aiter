// SPDX-License-Identifier: MIT
#include "mqa_logits.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { MQA_LOGITS_PYBIND; }
