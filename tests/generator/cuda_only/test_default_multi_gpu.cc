#include <cstring>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {

std::vector<float> ReadFloats(const std::shared_ptr<Tensor> &t) {
    Tensor cpu = t->To(Device(Device::DeviceType::kCPU, 0));
    std::vector<float> out(cpu.NumElements());
    std::memcpy(out.data(), cpu.DataPtr(), out.size() * sizeof(float));
    return out;
}

} // namespace

TEST(GeneratorDefaultMultiGpu, Cuda0AndCuda1HaveIndependentDefaults) {
    REQUIRE_MIN_DEVICES(2);

    Device d0(Device::DeviceType::kCUDA, 0);
    Device d1(Device::DeviceType::kCUDA, 1);

    manual_seed(2026);
    auto &g0 = default_generator(d0);
    auto &g1 = default_generator(d1);
    EXPECT_NE(g0.impl().get(), g1.impl().get());
    EXPECT_EQ(g0.InitialSeed(), 2026u);
    EXPECT_EQ(g1.InitialSeed(), 2026u);

    auto t0 = std::make_shared<Tensor>(std::vector<int64_t>{16}, DataType::kFLOAT32, d0);
    auto t1 = std::make_shared<Tensor>(std::vector<int64_t>{16}, DataType::kFLOAT32, d1);
    auto impl0 = ResolveGenerator(std::nullopt, d0);
    auto impl1 = ResolveGenerator(std::nullopt, d1);
    Dispatcher::Instance().Call<void>({d0.type(), "UniformRandom"}, t0, 0.0f, 1.0f, impl0.get());
    Dispatcher::Instance().Call<void>({d1.type(), "UniformRandom"}, t1, 0.0f, 1.0f, impl1.get());

    EXPECT_EQ(ReadFloats(t0), ReadFloats(t1));

    Dispatcher::Instance().Call<void>({d0.type(), "UniformRandom"}, t0, 0.0f, 1.0f, impl0.get());
    Dispatcher::Instance().Call<void>({d1.type(), "UniformRandom"}, t1, 0.0f, 1.0f, impl1.get());
    EXPECT_EQ(ReadFloats(t0), ReadFloats(t1));

    g1.ManualSeed(99);
    Dispatcher::Instance().Call<void>({d1.type(), "UniformRandom"}, t1, 0.0f, 1.0f, impl1.get());
    Dispatcher::Instance().Call<void>({d0.type(), "UniformRandom"}, t0, 0.0f, 1.0f, impl0.get());
    EXPECT_NE(ReadFloats(t0), ReadFloats(t1));
}
