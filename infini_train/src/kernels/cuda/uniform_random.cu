#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

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

__global__ void UniformRandomKernel(float *data, size_t n, float lo, float scale, uint64_t seed,
                                    uint64_t offset_4tuple) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    // Each thread owns a distinct subsequence (= tid); offset_4tuple is shared.
    infini_train::core::cuda::Philox4_32 eng(seed, /*subsequence=*/tid, /*offset=*/offset_4tuple * 4);
    for (size_t i = tid; i < n; i += stride) {
        data[i] = lo + eng.Uniform01() * scale;
    }
}

} // namespace

void UniformRandom(std::shared_ptr<Tensor> tensor, float lo, float hi, core::GeneratorImpl *impl) {
    CHECK(impl != nullptr) << "UniformRandom: GeneratorImpl is null";
    CHECK_EQ(static_cast<int>(tensor->Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "CUDA UniformRandom currently only supports FP32";
    CHECK(impl->device().IsCUDA()) << "CUDA UniformRandom got non-CUDA Generator";

    auto *cuda_impl = static_cast<core::CUDAGeneratorImpl *>(impl);
    const auto device = tensor->GetDevice();
    core::DeviceGuard guard(device);

    const size_t n = static_cast<size_t>(tensor->NumElements());
    if (n == 0) {
        return;
    }

    // Host-side mutex + Philox state advance (spec §5.4).
    core::CUDAGeneratorImpl::PhiloxState ph;
    {
        std::lock_guard<std::mutex> lk(cuda_impl->mutex());
        ph = cuda_impl->NextPhiloxState(static_cast<uint64_t>(n));
    }

    const auto cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                 infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                 ->cuda_stream();

    constexpr int kThreadsPerBlock = 256;
    const int num_blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    const float scale = hi - lo;
    UniformRandomKernel<<<num_blocks, kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<float *>(tensor->DataPtr()), n, lo, scale, ph.seed, ph.offset);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infini_train::kernels::cuda

REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, UniformRandom,
                infini_train::kernels::cuda::UniformRandom)
