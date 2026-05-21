#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/modules/module.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {

// 一个最小可挂子模块的 Module，用 modules_["x"] 暴露子节点。
class Bag : public nn::Module {
public:
    explicit Bag(const std::string &type = "Bag") : nn::Module(type) {}
    void Add(const std::string &name, std::shared_ptr<nn::Module> child) {
        modules_[name] = std::move(child);
    }
};

class Leaf : public nn::Module {
public:
    Leaf() : nn::Module("Leaf") {}
};

} // namespace

class ModuleTrainingTest : public infini_train::test::InfiniTrainTest {};

TEST_P(ModuleTrainingTest, DefaultIsTrainingTrue) {
    Leaf m;
    EXPECT_TRUE(m.IsTraining());
}

TEST_P(ModuleTrainingTest, EvalRecursesIntoChildren) {
    auto root = std::make_shared<Bag>("Root");
    auto leaf1 = std::make_shared<Leaf>();
    auto leaf2 = std::make_shared<Leaf>();
    root->Add("a", leaf1);
    root->Add("b", leaf2);

    root->Eval();
    EXPECT_FALSE(root->IsTraining());
    EXPECT_FALSE(leaf1->IsTraining());
    EXPECT_FALSE(leaf2->IsTraining());

    root->Train();
    EXPECT_TRUE(root->IsTraining());
    EXPECT_TRUE(leaf1->IsTraining());
    EXPECT_TRUE(leaf2->IsTraining());
}

TEST_P(ModuleTrainingTest, RecursesIntoPpPrefixedChildren) {
    // spec §5.0 关键不变量：__pp_* 子模块不跳过递归（与 StateDict 跳过逻辑相反）。
    auto root = std::make_shared<Bag>("Root");
    auto pp = std::make_shared<Leaf>();
    auto normal = std::make_shared<Leaf>();
    root->Add("__pp_chunk_0", pp);
    root->Add("attn", normal);

    root->Eval();
    EXPECT_FALSE(pp->IsTraining()) << "__pp_* must NOT be skipped by Train()";
    EXPECT_FALSE(normal->IsTraining());
}

TEST_P(ModuleTrainingTest, NullChildSkipped) {
    // 对齐 NamedModules:113 的 nullptr 容忍。
    auto root = std::make_shared<Bag>("Root");
    root->Add("missing", nullptr);
    EXPECT_NO_THROW(root->Eval());
    EXPECT_FALSE(root->IsTraining());
}

INFINI_TRAIN_REGISTER_TEST(ModuleTrainingTest);
