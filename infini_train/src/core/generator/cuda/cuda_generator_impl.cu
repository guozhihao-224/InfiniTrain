#include "infini_train/include/core/generator/cuda_generator_impl.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/generator/generator_impl.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

CUDAGeneratorImpl::CUDAGeneratorImpl(int8_t device_index)
    : GeneratorImpl(Device(Device::DeviceType::kCUDA, device_index)) {
    CHECK(0 <= device_index && device_index < kMaxGpus)
        << "CUDA Generator device index out of range: " << static_cast<int>(device_index);
}

void CUDAGeneratorImpl::SetCurrentSeed(uint64_t seed) {
    initial_seed_ = seed;
    seed_ = seed;
    offset_ = 0;
}

void CUDAGeneratorImpl::Reseed(uint64_t seed) {
    seed_ = seed;
    offset_ = 0;
}

CUDAGeneratorImpl::PhiloxState CUDAGeneratorImpl::NextPhiloxState(uint64_t num_elements) {
    const PhiloxState captured{seed_, offset_};
    const uint64_t consumed_4tuples = (num_elements + 3) / 4;
    offset_ += consumed_4tuples;
    return captured;
}

// Task 4 will replace these with magic/version-headed serialization.
std::vector<uint8_t> CUDAGeneratorImpl::GetState() const {
    LOG(FATAL) << "CUDAGeneratorImpl::GetState() not implemented yet (Task 4)";
    return {};
}

void CUDAGeneratorImpl::SetState(const std::vector<uint8_t> & /*state*/) {
    LOG(FATAL) << "CUDAGeneratorImpl::SetState() not implemented yet (Task 4)";
}

namespace {

std::shared_ptr<GeneratorImpl> CudaFactory(int8_t device_index) {
    return std::make_shared<CUDAGeneratorImpl>(device_index);
}

} // namespace

} // namespace infini_train::core

INFINI_TRAIN_REGISTER_GENERATOR_IMPL(kCUDA, ::infini_train::core::CudaFactory)
