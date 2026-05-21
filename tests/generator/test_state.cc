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

class GeneratorStateTest : public infini_train::test::InfiniTrainTest {};

namespace {
std::vector<float> ReadCpuFloats(const std::shared_ptr<Tensor> &t) {
    Tensor cpu = t->To(Device(Device::DeviceType::kCPU, 0));
    std::vector<float> out(cpu.NumElements());
    std::memcpy(out.data(), cpu.DataPtr(), out.size() * sizeof(float));
    return out;
}

std::vector<float> DrawUniform(Generator &g, int64_t n) {
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{n}, DataType::kFLOAT32, g.device());
    Dispatcher::Instance().Call<void>({g.device().type(), "UniformRandom"}, t, 0.0f, 1.0f, g.impl().get());
    return ReadCpuFloats(t);
}
} // namespace

TEST_P(GeneratorStateTest, GetSetStateRoundtrip) {
    Generator gen(GetDevice());
    gen.ManualSeed(2026);

    const auto baseline = gen.GetState();
    gen.ManualSeed(99);
    EXPECT_NE(gen.GetState(), baseline);
    gen.SetState(baseline);
    EXPECT_EQ(gen.GetState(), baseline);
    EXPECT_EQ(gen.InitialSeed(), 2026u);
}

// Spec §三(3): save state, draw a sequence, restore the state, draw again,
// and confirm the second draw aligns with the original sequence. Verifies
// the GetState/SetState contract end-to-end at the random-number layer
// rather than only at the state-blob layer.
TEST_P(GeneratorStateTest, GetSetStateRestoresSequence) {
    Generator gen(GetDevice());
    gen.ManualSeed(2026);

    const auto warmup = DrawUniform(gen, 8);
    const auto checkpoint = gen.GetState();
    const auto seq_first = DrawUniform(gen, 8);

    EXPECT_NE(warmup, seq_first);

    gen.SetState(checkpoint);
    const auto seq_replayed = DrawUniform(gen, 8);
    EXPECT_EQ(seq_first, seq_replayed);
}

INFINI_TRAIN_REGISTER_TEST(GeneratorStateTest);
