#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/core/generator/cpu_generator_impl.h"
#include "infini_train/include/core/generator/cuda_generator_impl.h"

using infini_train::core::CPUGeneratorImpl;
using infini_train::core::CUDAGeneratorImpl;

namespace {

constexpr uint32_t kCudaMagic = 0x47445543;   // 'CUDG'
constexpr uint32_t kCudaVersion = 1;

std::vector<uint8_t> MakeHeader(uint32_t magic, uint32_t version, uint64_t payload_size) {
    std::vector<uint8_t> buf(16);
    std::memcpy(buf.data() + 0, &magic, 4);
    std::memcpy(buf.data() + 4, &version, 4);
    std::memcpy(buf.data() + 8, &payload_size, 8);
    return buf;
}

}  // namespace

TEST(CudaStateValidation, RoundtripPreservesSeedOffsetInitialSeed) {
    CUDAGeneratorImpl impl(0);
    impl.SetCurrentSeed(2026);
    impl.NextPhiloxState(13);   // 把 offset 推进到 4
    const auto blob = impl.GetState();

    CUDAGeneratorImpl other(0);
    other.SetCurrentSeed(1);    // 故意弄脏
    other.SetState(blob);
    EXPECT_EQ(other.CurrentSeed(), 2026u);
    EXPECT_EQ(other.InitialSeed(), 2026u);
    auto ph = other.NextPhiloxState(0);
    EXPECT_EQ(ph.offset, 4u);
}

TEST(CudaStateValidation, TruncatedHeader) {
    CUDAGeneratorImpl impl(0);
    impl.SetCurrentSeed(1);
    std::vector<uint8_t> too_short(15, 0);
    EXPECT_THROW(impl.SetState(too_short), std::runtime_error);
}

TEST(CudaStateValidation, BadMagic) {
    CUDAGeneratorImpl impl(0);
    impl.SetCurrentSeed(1);
    auto buf = MakeHeader(0xDEADBEEF, kCudaVersion, 0);
    EXPECT_THROW(impl.SetState(buf), std::runtime_error);
}

TEST(CudaStateValidation, BadVersion) {
    CUDAGeneratorImpl impl(0);
    impl.SetCurrentSeed(1);
    auto buf = MakeHeader(kCudaMagic, kCudaVersion + 1, 0);
    EXPECT_THROW(impl.SetState(buf), std::runtime_error);
}

TEST(CudaStateValidation, PayloadSizeMismatch) {
    CUDAGeneratorImpl impl(0);
    impl.SetCurrentSeed(1);
    auto buf = MakeHeader(kCudaMagic, kCudaVersion, 100);
    buf.insert(buf.end(), {0xAA, 0xBB, 0xCC, 0xDD});
    EXPECT_THROW(impl.SetState(buf), std::runtime_error);
}

TEST(CudaStateValidation, RejectsCpuStateBlob) {
    // spec §7.2: CPU GetState() 喂给 CUDA SetState() 因 magic 不匹配抛。
    CPUGeneratorImpl cpu(0);
    cpu.SetCurrentSeed(7);
    auto cpu_blob = cpu.GetState();

    CUDAGeneratorImpl cuda(0);
    cuda.SetCurrentSeed(7);
    EXPECT_THROW(cuda.SetState(cpu_blob), std::runtime_error);
}
