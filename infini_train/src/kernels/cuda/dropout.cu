#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <tuple>

#include <cuda_runtime.h>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/generator/cuda_generator_impl.h"
#include "infini_train/include/core/generator/generator_impl.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/generator/cuda/philox_engine.cuh"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

namespace {

__global__ void DropoutForwardKernel(const float *in, float *out, uint8_t *mask, size_t n, float p, float scale,
                                     uint64_t seed, uint64_t offset_4tuple) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    // Each thread owns a distinct subsequence (= tid); offset_4tuple is shared.
    infini_train::core::cuda::Philox4_32 eng(seed, /*subsequence=*/tid, /*offset=*/offset_4tuple * 4);
    for (size_t i = tid; i < n; i += stride) {
        const float r = eng.Uniform01();
        const bool keep = r >= p;
        mask[i] = keep ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
        out[i] = keep ? in[i] * scale : 0.0f;
    }
}

__global__ void DropoutBackwardKernel(const float *go, const uint8_t *mask, float *gi, size_t n, float scale) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = tid; i < n; i += stride) {
        gi[i] = mask[i] ? go[i] * scale : 0.0f;
    }
}

} // namespace

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
DropoutForward(const std::shared_ptr<Tensor> &input, float p, core::GeneratorImpl *impl) {
    CHECK(input != nullptr);
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "CUDA DropoutForward only supports FP32";
    CHECK_GE(p, 0.0f);
    CHECK_LT(p, 1.0f) << "Dropout p must be in [0, 1)";

    const auto device = input->GetDevice();
    core::DeviceGuard guard(device);

    const auto &shape = input->Dims();
    auto output = std::make_shared<Tensor>(shape, DataType::kFLOAT32, device);
    auto mask = std::make_shared<Tensor>(shape, DataType::kUINT8, device);
    const size_t n = static_cast<size_t>(input->NumElements());
    if (n == 0) {
        return {output, mask};
    }

    const auto cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                 infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                 ->cuda_stream();

    if (p == 0.0f) {
        // keep-all fast path: no RNG, just copy input -> output and set mask to 1.
        CUDA_CHECK(cudaMemcpyAsync(output->DataPtr(), input->DataPtr(), n * sizeof(float),
                                   cudaMemcpyDeviceToDevice, cuda_stream));
        CUDA_CHECK(cudaMemsetAsync(mask->DataPtr(), 1, n, cuda_stream));
        return {output, mask};
    }

    CHECK(impl != nullptr) << "DropoutForward: GeneratorImpl is null";
    CHECK(impl->device().IsCUDA()) << "CUDA DropoutForward got non-CUDA Generator";
    auto *cuda_impl = static_cast<core::CUDAGeneratorImpl *>(impl);

    // Host-side mutex + Philox state advance (spec §5.4); release the lock
    // before the kernel launch since the launch only consumes the snapshot.
    core::CUDAGeneratorImpl::PhiloxState ph;
    {
        std::lock_guard<std::mutex> lk(cuda_impl->mutex());
        ph = cuda_impl->NextPhiloxState(static_cast<uint64_t>(n));
    }

    constexpr int kThreadsPerBlock = 256;
    const int num_blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    const float scale = 1.0f / (1.0f - p);
    DropoutForwardKernel<<<num_blocks, kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<float *>(output->DataPtr()),
        static_cast<uint8_t *>(mask->DataPtr()), n, p, scale, ph.seed, ph.offset);
    CUDA_CHECK(cudaGetLastError());
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

    const auto device = grad_output->GetDevice();
    core::DeviceGuard guard(device);

    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), DataType::kFLOAT32, device);
    const size_t n = static_cast<size_t>(grad_output->NumElements());
    if (n == 0) {
        return grad_input;
    }

    const auto cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                 infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                 ->cuda_stream();

    constexpr int kThreadsPerBlock = 256;
    const int num_blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    const float scale = 1.0f / (1.0f - p);
    DropoutBackwardKernel<<<num_blocks, kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<const float *>(grad_output->DataPtr()), static_cast<const uint8_t *>(mask->DataPtr()),
        static_cast<float *>(grad_input->DataPtr()), n, scale);
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

} // namespace infini_train::kernels::cuda

REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, DropoutForward, infini_train::kernels::cuda::DropoutForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, DropoutBackward, infini_train::kernels::cuda::DropoutBackward)
