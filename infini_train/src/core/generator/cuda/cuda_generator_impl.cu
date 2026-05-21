#include "infini_train/include/core/generator/cuda_generator_impl.h"

#include <cstdint>
#include <cstring>
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

namespace {

inline void WriteBytes(std::vector<uint8_t> &dst, const void *src, size_t n) {
    const auto *p = static_cast<const uint8_t *>(src);
    dst.insert(dst.end(), p, p + n);
}

inline size_t ReadBytes(const std::vector<uint8_t> &src, size_t offset, void *dst, size_t n) {
    if (offset + n > src.size()) {
        throw std::runtime_error("CUDAGeneratorImpl::SetState: buffer underflow");
    }
    std::memcpy(dst, src.data() + offset, n);
    return offset + n;
}

} // namespace

std::vector<uint8_t> CUDAGeneratorImpl::GetState() const {
    std::vector<uint8_t> payload;
    payload.reserve(8 + 8 + 8);
    WriteBytes(payload, &seed_, 8);
    WriteBytes(payload, &offset_, 8);
    WriteBytes(payload, &initial_seed_, 8);

    const uint64_t payload_size = payload.size();
    std::vector<uint8_t> out;
    out.reserve(16 + payload_size);
    const uint32_t magic = kMagic;
    const uint32_t version = kVersion;
    WriteBytes(out, &magic, 4);
    WriteBytes(out, &version, 4);
    WriteBytes(out, &payload_size, 8);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

void CUDAGeneratorImpl::SetState(const std::vector<uint8_t> &state) {
    if (state.size() < 16) {
        throw std::runtime_error("CUDAGeneratorImpl::SetState: header truncated");
    }
    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t payload_size = 0;
    size_t off = 0;
    off = ReadBytes(state, off, &magic, 4);
    off = ReadBytes(state, off, &version, 4);
    off = ReadBytes(state, off, &payload_size, 8);
    if (magic != kMagic) {
        throw std::runtime_error("CUDAGeneratorImpl::SetState: magic mismatch");
    }
    if (version != kVersion) {
        throw std::runtime_error("CUDAGeneratorImpl::SetState: version mismatch");
    }
    if (off + payload_size != state.size()) {
        throw std::runtime_error("CUDAGeneratorImpl::SetState: payload size mismatch");
    }

    uint64_t new_seed = 0;
    uint64_t new_offset = 0;
    uint64_t new_initial = 0;
    off = ReadBytes(state, off, &new_seed, 8);
    off = ReadBytes(state, off, &new_offset, 8);
    off = ReadBytes(state, off, &new_initial, 8);
    seed_ = new_seed;
    offset_ = new_offset;
    initial_seed_ = new_initial;
}

namespace {

std::shared_ptr<GeneratorImpl> CudaFactory(int8_t device_index) {
    return std::make_shared<CUDAGeneratorImpl>(device_index);
}

} // namespace

} // namespace infini_train::core

INFINI_TRAIN_REGISTER_GENERATOR_IMPL(kCUDA, ::infini_train::core::CudaFactory)
