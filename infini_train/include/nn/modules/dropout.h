#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "infini_train/include/generator.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::nn {

class Dropout : public Module {
public:
    static constexpr char kType[] = "Dropout";

    explicit Dropout(float p = 0.5f, std::optional<Generator> generator = std::nullopt);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    float p_;
    std::optional<Generator> generator_;
};

} // namespace infini_train::nn
