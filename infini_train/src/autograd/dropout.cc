#include "infini_train/include/autograd/dropout.h"

#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>>
Dropout::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1u);
    const auto &input = input_tensors[0];
    const auto device = input->GetDevice();
    auto impl = ResolveGenerator(generator_, device);

    auto [output, mask]
        = Dispatcher::Instance().Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
              {device.type(), "DropoutForward"}, input, p_, impl.get());

    saved_tensors_ = {mask};
    return {output};
}

std::vector<std::shared_ptr<Tensor>>
Dropout::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1u);
    CHECK_EQ(saved_tensors_.size(), 1u);
    const auto &grad_output = grad_outputs[0];
    const auto &mask = saved_tensors_[0];

    auto grad_input = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {grad_output->GetDevice().type(), "DropoutBackward"}, grad_output, mask, p_);
    return {grad_input};
}

} // namespace infini_train::autograd
