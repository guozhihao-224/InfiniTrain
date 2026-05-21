#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/generator/cpu_generator_impl.h"
#include "infini_train/include/core/generator/generator_impl.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
DropoutForward(const std::shared_ptr<Tensor> &input, float p, core::GeneratorImpl *impl) {
    CHECK(input != nullptr);
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "CPU DropoutForward only supports FP32";
    CHECK_GE(p, 0.0f);
    CHECK_LT(p, 1.0f) << "Dropout p must be in [0, 1)";

    const auto &shape = input->Dims();
    const auto device = input->GetDevice();
    auto output = std::make_shared<Tensor>(shape, DataType::kFLOAT32, device);
    auto mask = std::make_shared<Tensor>(shape, DataType::kUINT8, device);

    const size_t n = static_cast<size_t>(input->NumElements());
    const auto *in = static_cast<const float *>(input->DataPtr());
    auto *out = static_cast<float *>(output->DataPtr());
    auto *m = static_cast<uint8_t *>(mask->DataPtr());

    if (p == 0.0f) {
        // keep-all
        for (size_t i = 0; i < n; ++i) {
            out[i] = in[i];
            m[i] = 1;
        }
        return {output, mask};
    }

    CHECK(impl != nullptr) << "DropoutForward: GeneratorImpl is null";
    auto *cpu_impl = static_cast<core::CPUGeneratorImpl *>(impl);
    std::lock_guard<std::mutex> lk(cpu_impl->mutex());
    auto &eng = cpu_impl->engine();
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float scale = 1.0f / (1.0f - p);

    for (size_t i = 0; i < n; ++i) {
        const float r = dist(eng);
        const bool keep = r >= p;
        m[i] = keep ? 1 : 0;
        out[i] = keep ? in[i] * scale : 0.0f;
    }
    return {output, mask};
}

std::shared_ptr<Tensor> DropoutBackward(const std::shared_ptr<Tensor> &grad_output,
                                        const std::shared_ptr<Tensor> &mask, float p) {
    CHECK(grad_output != nullptr);
    CHECK(mask != nullptr);
    CHECK_EQ(static_cast<int>(grad_output->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(mask->Dtype()), static_cast<int>(DataType::kUINT8));
    CHECK_EQ(grad_output->NumElements(), mask->NumElements());
    CHECK_GE(p, 0.0f);
    CHECK_LT(p, 1.0f);

    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), DataType::kFLOAT32, grad_output->GetDevice());
    const size_t n = static_cast<size_t>(grad_output->NumElements());
    const auto *go = static_cast<const float *>(grad_output->DataPtr());
    const auto *m = static_cast<const uint8_t *>(mask->DataPtr());
    auto *gi = static_cast<float *>(grad_input->DataPtr());
    const float scale = 1.0f / (1.0f - p);

    for (size_t i = 0; i < n; ++i) { gi[i] = m[i] ? go[i] * scale : 0.0f; }
    return grad_input;
}

} // namespace infini_train::kernels::cpu

REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, DropoutForward, infini_train::kernels::cpu::DropoutForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, DropoutBackward, infini_train::kernels::cpu::DropoutBackward)
