#pragma once

#ifndef USE_CUDA
#error "cuda_generator_impl.h requires USE_CUDA=ON"
#endif

#include <cstdint>
#include <vector>

#include "infini_train/include/core/generator/generator_impl.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

class CUDAGeneratorImpl : public GeneratorImpl {
public:
    explicit CUDAGeneratorImpl(int8_t device_index);

    void SetCurrentSeed(uint64_t seed) override;
    uint64_t CurrentSeed() const override { return seed_; }
    uint64_t InitialSeed() const override { return initial_seed_; }

    std::vector<uint8_t> GetState() const override;
    void SetState(const std::vector<uint8_t> &state) override;
    uint32_t StateMagic() const override { return kMagic; }

    struct PhiloxState {
        uint64_t seed;
        uint64_t offset;
    };

    // Advance offset: consume num_elements randoms, 4 per Philox 4-tuple.
    // Caller holds mutex().
    PhiloxState NextPhiloxState(uint64_t num_elements);

protected:
    void Reseed(uint64_t seed) override;

private:
    static constexpr uint32_t kMagic = 0x47445543; // 'CUDG'
    static constexpr uint32_t kVersion = 1;

    uint64_t initial_seed_{0};
    uint64_t seed_{0};
    uint64_t offset_{0};
};

} // namespace infini_train::core
