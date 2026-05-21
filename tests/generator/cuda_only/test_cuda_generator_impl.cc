#include <memory>

#include "gtest/gtest.h"

#include "infini_train/include/core/generator/cuda_generator_impl.h"

using infini_train::core::CUDAGeneratorImpl;

TEST(CUDAGeneratorImplTest, ManualSeedSetsCurrentInitialAndZeroOffset) {
    CUDAGeneratorImpl impl(0);
    impl.SetCurrentSeed(42);
    EXPECT_EQ(impl.CurrentSeed(), 42u);
    EXPECT_EQ(impl.InitialSeed(), 42u);
    auto ph = impl.NextPhiloxState(0);
    EXPECT_EQ(ph.seed, 42u);
    EXPECT_EQ(ph.offset, 0u);
}

TEST(CUDAGeneratorImplTest, NextPhiloxStateAdvancesOffset) {
    CUDAGeneratorImpl impl(0);
    impl.SetCurrentSeed(1);
    auto p1 = impl.NextPhiloxState(7);          // ceil(7/4) = 2
    EXPECT_EQ(p1.offset, 0u);
    auto p2 = impl.NextPhiloxState(9);          // ceil(9/4) = 3
    EXPECT_EQ(p2.offset, 2u);
    auto p3 = impl.NextPhiloxState(0);
    EXPECT_EQ(p3.offset, 5u);                   // 2 + 3
}

TEST(CUDAGeneratorImplTest, ReseedThroughBaseClassSeed) {
    CUDAGeneratorImpl impl(0);
    impl.SetCurrentSeed(99);
    const uint64_t initial_before = impl.InitialSeed();
    const uint64_t s = impl.Seed();
    EXPECT_EQ(impl.CurrentSeed(), s);
    EXPECT_EQ(impl.InitialSeed(), initial_before);   // §4.3 invariant
    auto ph = impl.NextPhiloxState(0);
    EXPECT_EQ(ph.offset, 0u);                        // Reseed zeroes offset
}

TEST(CUDAGeneratorImplTest, IndexOutOfRangeAborts) {
    EXPECT_DEATH(CUDAGeneratorImpl(8), "device index out of range");
    EXPECT_DEATH(CUDAGeneratorImpl(-1), "device index out of range");
}
