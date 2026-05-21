#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include <cmath>
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

__device__ inline void BoxMuller(float u1, float u2, float &z0, float &z1) {
    // Avoid log(0).
    constexpr float kEps = 1.1754944e-38f; // FLT_MIN
    const float r = sqrtf(-2.0f * logf(fmaxf(u1, kEps)));
    float s, c;
    sincosf(6.2831853071795864769f * u2, &s, &c);
    z0 = r * c;
    z1 = r * s;
}

__global__ void NormalRandomKernel(float *data, size_t n, float mean, float std, uint64_t seed,
                                   uint64_t offset_4tuple) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    infini_train::core::cuda::Philox4_32 eng(seed, /*subsequence=*/tid, /*offset=*/offset_4tuple * 4);

    // Each iteration writes (z0, z1); if n is odd we write z0 only on the final pair.
    for (size_t base = tid * 2; base < n; base += stride * 2) {
        const float u1 = eng.Uniform01();
        const float u2 = eng.Uniform01();
        float z0;
        float z1;
        BoxMuller(u1, u2, z0, z1);
        data[base] = mean + std * z0;
        if (base + 1 < n) {
            data[base + 1] = mean + std * z1;
        }
    }
}

} // namespace

void NormalRandom(std::shared_ptr<Tensor> tensor, float mean, float std, core::GeneratorImpl *impl) {
    CHECK(impl != nullptr) << "NormalRandom: GeneratorImpl is null";
    CHECK_EQ(static_cast<int>(tensor->Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "CUDA NormalRandom currently only supports FP32";
    CHECK(impl->device().IsCUDA()) << "CUDA NormalRandom got non-CUDA Generator";

    auto *cuda_impl = static_cast<core::CUDAGeneratorImpl *>(impl);
    const auto device = tensor->GetDevice();
    core::DeviceGuard guard(device);

    const size_t n = static_cast<size_t>(tensor->NumElements());
    if (n == 0) {
        return;
    }

    core::CUDAGeneratorImpl::PhiloxState ph;
    {
        std::lock_guard<std::mutex> lk(cuda_impl->mutex());
        // Box-Muller consumes 2 uniforms per element; use a simple upper bound
        // of 2*n. Replay-correctness only requires this rule to stay stable.
        ph = cuda_impl->NextPhiloxState(2 * static_cast<uint64_t>(n));
    }

    const auto cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                 infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                 ->cuda_stream();

    constexpr int kThreadsPerBlock = 256;
    // Each thread writes 2 elements per iteration, so grid = ceil(n / 2 / threadsPerBlock).
    const size_t pairs = (n + 1) / 2;
    const int num_blocks = static_cast<int>((pairs + kThreadsPerBlock - 1) / kThreadsPerBlock);
    NormalRandomKernel<<<num_blocks, kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<float *>(tensor->DataPtr()), n, mean, std, ph.seed, ph.offset);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infini_train::kernels::cuda

REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, NormalRandom,
                infini_train::kernels::cuda::NormalRandom)
