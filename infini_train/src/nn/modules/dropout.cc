#include "infini_train/include/nn/modules/dropout.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/dropout.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

Dropout::Dropout(float p, std::optional<Generator> generator)
    : Module(kType), p_(p), generator_(std::move(generator)) {
    CHECK_GE(p_, 0.0f);
    CHECK_LT(p_, 1.0f) << "Dropout p must be in [0, 1)";
}

std::vector<std::shared_ptr<Tensor>>
Dropout::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1u);
    if (!IsTraining() || p_ == 0.0f) {
        return input_tensors;
    }
    auto fn = std::make_shared<autograd::Dropout>(p_, generator_);
    return fn->Apply(input_tensors);
}

} // namespace infini_train::nn
