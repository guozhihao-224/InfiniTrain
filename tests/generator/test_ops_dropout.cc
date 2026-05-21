#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/dropout.h"
#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/dropout.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorDropoutOpTest : public infini_train::test::InfiniTrainTest {};

namespace {

std::vector<float> ReadFloats(const std::shared_ptr<Tensor> &t) {
    Tensor cpu = t->To(Device(Device::DeviceType::kCPU, 0));
    std::vector<float> out(cpu.NumElements());
    std::memcpy(out.data(), cpu.DataPtr(), out.size() * sizeof(float));
    return out;
}

std::vector<uint8_t> ReadBytes(const std::shared_ptr<Tensor> &t) {
    Tensor cpu = t->To(Device(Device::DeviceType::kCPU, 0));
    std::vector<uint8_t> out(cpu.NumElements());
    std::memcpy(out.data(), cpu.DataPtr(), out.size());
    return out;
}

std::shared_ptr<Tensor> MakeOnes(const std::vector<int64_t> &shape, Device device) {
    // Build on CPU then move; avoids needing a device-side fill helper here.
    Tensor cpu(shape, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    auto *p = static_cast<float *>(cpu.DataPtr());
    for (int64_t i = 0; i < cpu.NumElements(); ++i) p[i] = 1.0f;
    auto t = std::make_shared<Tensor>(cpu.To(device));
    return t;
}

} // namespace

TEST_P(GeneratorDropoutOpTest, SameSeedSameMask) {
    auto x = MakeOnes({64}, GetDevice());

    Generator g1(GetDevice());
    Generator g2(GetDevice());
    g1.ManualSeed(2026);
    g2.ManualSeed(2026);

    auto fn1 = std::make_shared<autograd::Dropout>(0.5f, std::optional<Generator>{g1});
    auto fn2 = std::make_shared<autograd::Dropout>(0.5f, std::optional<Generator>{g2});
    auto y1 = fn1->Apply({x})[0];
    auto y2 = fn2->Apply({x})[0];

    EXPECT_EQ(ReadFloats(y1), ReadFloats(y2));
}

TEST_P(GeneratorDropoutOpTest, MaskKeepRateSane) {
    constexpr int64_t kN = 1 << 14; // 16384 elements
    auto x = MakeOnes({kN}, GetDevice());

    Generator g(GetDevice());
    g.ManualSeed(7);
    const float p = 0.3f;
    auto fn = std::make_shared<autograd::Dropout>(p, std::optional<Generator>{g});
    auto y = fn->Apply({x})[0];

    // y == 0 iff the element was dropped; count keeps via y != 0.
    auto vals = ReadFloats(y);
    int kept = 0;
    for (float v : vals) {
        if (v != 0.0f) ++kept;
    }
    const float observed = static_cast<float>(kept) / static_cast<float>(kN);
    const float expected = 1.0f - p;
    EXPECT_NEAR(observed, expected, 0.05f) << "observed keep-rate " << observed << " expected ~" << expected;
}

TEST_P(GeneratorDropoutOpTest, OutputScaleCorrect) {
    // For x_i = 1, kept y_i must equal 1/(1-p), dropped y_i must equal 0.
    auto x = MakeOnes({256}, GetDevice());

    Generator g(GetDevice());
    g.ManualSeed(11);
    const float p = 0.4f;
    const float scale = 1.0f / (1.0f - p);
    auto fn = std::make_shared<autograd::Dropout>(p, std::optional<Generator>{g});
    auto y = fn->Apply({x})[0];

    auto vals = ReadFloats(y);
    for (float v : vals) {
        const bool ok = (v == 0.0f) || (std::fabs(v - scale) < 1e-6f);
        EXPECT_TRUE(ok) << "unexpected dropout output: " << v;
    }
}

TEST_P(GeneratorDropoutOpTest, BackwardEqualsMaskScale) {
    // grad_input == grad_output * mask * (1/(1-p)). With grad_output = 1, kept slots
    // get scale, dropped slots get 0 -- which equals y itself when input is ones.
    auto x = MakeOnes({256}, GetDevice());

    Generator g(GetDevice());
    g.ManualSeed(42);
    const float p = 0.25f;
    auto fn = std::make_shared<autograd::Dropout>(p, std::optional<Generator>{g});
    auto y = fn->Apply({x})[0];

    auto grad_out = MakeOnes({256}, GetDevice());
    auto grads = fn->Backward({grad_out});
    ASSERT_EQ(grads.size(), 1u);
    EXPECT_EQ(ReadFloats(grads[0]), ReadFloats(y));
}

TEST_P(GeneratorDropoutOpTest, EvalGateBypassesDropout) {
    auto x = MakeOnes({128}, GetDevice());
    auto module = std::make_shared<nn::Dropout>(0.5f);
    module->Eval();
    auto out = module->Forward({x})[0];
    EXPECT_EQ(ReadFloats(out), ReadFloats(x));
}

TEST_P(GeneratorDropoutOpTest, FunctionalTrainingFalseBypasses) {
    auto x = MakeOnes({128}, GetDevice());
    auto out = nn::function::Dropout(x, /*p=*/0.5f, /*training=*/false);
    EXPECT_EQ(ReadFloats(out), ReadFloats(x));
}

INFINI_TRAIN_REGISTER_TEST(GeneratorDropoutOpTest);
