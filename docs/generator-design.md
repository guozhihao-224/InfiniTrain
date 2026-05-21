# InfiniTrain Generator 抽象 — 技术报告

> 2026 春季人工智能大赛 Generator 选题。提交 PR：`guozhihao-224:feat/generator-phase1` → `InfiniTensor:master`。
>
> 分支 `feat/generator-phase1`，HEAD `a44c755`，自 `master` 起 37 commits（14 Phase 1 / 10 Phase 2 / 7 Phase 3 / 6 Phase 4）。
>
> 设计 spec：[`docs/superpowers/specs/2026-05-19-generator-abstraction-design.md`](superpowers/specs/2026-05-19-generator-abstraction-design.md)、[`2026-05-21-generator-phase3-dropout-design.md`](superpowers/specs/2026-05-21-generator-phase3-dropout-design.md)；实测留档：[`docs/superpowers/notes/2026-05-21-phase2-pr-prep.md`](superpowers/notes/2026-05-21-phase2-pr-prep.md)。
>
> 本报告对应赛题 §四《提交报告要求》三个小节：§1 功能正确性验证、§2 对齐性与行为分析报告、§3 测试与可复现性说明；§4 给出 §五《验收要求》逐条命中清单。

---

## 0. 项目概览

**实现的内容（按赛题 §三 任务拆解）：**

| 赛题任务 | 命中实现 |
|---|---|
| §三(1) Generator 抽象与状态管理 | `Generator` PImpl 句柄 + `core::GeneratorImpl` 基类 + `GeneratorImplRegistry`（仿 `DeviceGuardImplRegistry`），统一 `ManualSeed/Seed/InitialSeed/GetState/SetState/device()`，状态头部 `magic(4)\|version(4)\|payload_size(8)`，`std::mutex` 线程安全保护 |
| §三(2) CPU + CUDA 后端 GeneratorImpl | `CPUGeneratorImpl`（`std::mt19937_64`，magic `'CPUG'`） + `CUDAGeneratorImpl`（vendored PyTorch Philox4_32 device engine，BSD-3 attribution，16-byte 状态 `seed+offset+initial_seed`，magic `'CUDG'`，`NextPhiloxState(num_elements)` 在 mutex 下推进 offset） |
| §三(3) 默认池 + 全局 seed 入口 | `default_generator(Device)`（CPU 单例 + 每 CUDA 设备独立，`std::once_flag` 惰性初始化）、`manual_seed(uint64_t)`（同步 CPU + 已/未初始化 CUDA 默认）、`manual_seed_cuda(uint64_t)`（仅 CUDA 默认）、`last_user_seed_` 记忆机制让"先 manual_seed 再首次取默认"有效 |
| §三(4) 算子接入改造 | 初始化类：`nn::init::Uniform/Normal/KaimingUniform`（`init.cc`） + `Tensor::Uniform`；训练期：`nn::Dropout` 模块 + `nn::function::Dropout/Rand/Randn` 自由函数。所有路径通过 `ResolveGenerator(std::optional<Generator>, Device)` 决定显式 vs 默认 |
| §三(5) 与 PyTorch 对齐验证 | 见本报告 §2.2 接口对照表 + §2.3 已对齐项；语义对齐项均由 `INFINI_TRAIN_REGISTER_TEST` 在 CPU+CUDA 双实例化测试覆盖 |

**额外完成（赛题 §三 未要求但与"统一随机行为"相关）：**

- `nn::Module::Train(bool)/Eval()/IsTraining()` 递归切换训练态（递归 `modules_`，**包括 `__pp_*` 前缀的 pipeline-parallel 子模块**，仅跳 `nullptr`）；`nn::Dropout` 由 `IsTraining()` 门控。
- 加分项 `scripts/check_reproducibility.sh`：自动跑两次 ctest 并 diff 每个测试 verdict，验证整体复现性。

**实测一句话总结（详见 §3 与 [PR-prep notes §10](superpowers/notes/2026-05-21-phase2-pr-prep.md)）：**

- Train-server CUDA Debug 完整 ctest：**510 pass / 10 fail / 12 skip / 532 ran**。
- Generator suite（`-R 'Generator|ModuleTrainingTest'`）：**68 pass / 0 fail / 1 skip**（多 GPU 测试在 1-GPU 主机自动 SKIP）。
- 10 fail 全部是 pre-existing 上游 `gemm.cu:41` DCHECK，**与本 PR 无关**——已在上游 master `a917760` 上复现同一组 10 个测试名失败（393 pass / 10 fail / 403 ran，详见 §3.5）。

---

## 1. 功能正确性验证

按赛题 §三《任务拆解》中"测试与对齐验证"的五类分别给出测试落点 + 通过状态。所有 `TEST_P` 通过 `INFINI_TRAIN_REGISTER_TEST` 自动 CPU+CUDA 双实例化（gtest 名分别为 `CPU/<Fixture>.<Case>` 与 `CUDA/<Fixture>.<Case>`）。

### 1.1 接口一致性（赛题 §三(1)）

| 验收点 | 测试用例 | 文件:行 |
|---|---|---|
| `Generator` 构造时绑定 device | `GeneratorBasicTest.ConstructionAttachesDevice` | `tests/generator/test_generator_basic.cc:14` |
| `ManualSeed` 同时刷新 current+initial | `GeneratorBasicTest.ManualSeedRoundtripsCurrentAndInitial` | `:20` |
| 句柄拷贝共享同一 Impl（PImpl 浅拷贝） | `GeneratorBasicTest.CopyShareImpl` | `:26` |
| 设备类型查询 / 默认 Generator 获取 | `GeneratorDefaultTest.DefaultGeneratorIsStable` | `tests/generator/test_default.cc:15` |
| 显式 Generator vs 默认两条调用路径 | `GeneratorDispatchTest.NullGeneratorFallsBackToDefault` | `tests/generator/test_dispatch.cc:18` |
| CPU Impl 直接接口（`SetCurrentSeed/Seed/InitialSeed`） | `CPUGeneratorImplTest.{ManualSeedSetsCurrentAndInitialSeed, EngineAdvancesAcrossCalls, SameManualSeedReproduces, SeedRandomizesCurrentButPreservesInitial, IndexNonZeroAborts}` | `tests/generator/cpu_only/test_cpu_generator_impl.cc:9-46` |
| CUDA Impl 直接接口 + offset 推进 | `CUDAGeneratorImplTest.{ManualSeedSetsCurrentInitialAndZeroOffset, NextPhiloxStateAdvancesOffset, ReseedThroughBaseClassSeed, IndexOutOfRangeAborts}` | `tests/generator/cuda_only/test_cuda_generator_impl.cc:9-45` |

**全部通过。**

### 1.2 种子可复现（赛题 §三(2)）

| 验收点 | 测试用例 | 文件:行 |
|---|---|---|
| 同 seed 多次 manual_seed 后 state 一致 | `GeneratorSeedTest.ManualSeedReseedsState` | `tests/generator/test_seed.cc:14` |
| 不同 seed 后 state 不同 | `GeneratorSeedTest.DifferentSeedsDifferState` | `:23` |
| `Seed()` 不改变 `InitialSeed()` | `GeneratorInitialSeedTest.SeedDoesNotChangeInitialSeed` | `tests/generator/test_initial_seed.cc:12` |
| 同 seed Uniform 输出一致 | `GeneratorUniformOpTest.SameSeedSameOutput` | `tests/generator/test_ops_uniform.cc:27` |
| 连续调用推进状态（state-advance trap） | `GeneratorUniformOpTest.ConsecutiveCallsAdvanceState` | `:40` |
| 输出落入 `[lo, hi)` | `GeneratorUniformOpTest.OutputsWithinRange` | `:50` |
| 同 seed Normal 输出一致 | `GeneratorNormalOpTest.SameSeedSameOutput` | `tests/generator/test_ops_normal.cc:28` |
| Normal 均值合理 | `GeneratorNormalOpTest.MeanCloseToTarget` | `:39` |
| 同 seed kaiming 权重一致、fan_in 边界 | `GeneratorKaimingTest.{SameSeedSameWeights, FanInBoundsRespected}` | `tests/generator/test_ops_kaiming.cc:28/41` |
| 同 seed Dropout mask 一致 | `GeneratorDropoutOpTest.SameSeedSameMask` | `tests/generator/test_ops_dropout.cc:49` |
| Dropout 保留率近 `1-p`（容差 ±0.05） | `GeneratorDropoutOpTest.MaskKeepRateSane` | `:65` |
| Dropout 输出 scale 正确（`x/(1-p)`） | `GeneratorDropoutOpTest.OutputScaleCorrect` | `:86` |

**全部通过**。`ConsecutiveCallsAdvanceState` 是 trap test：旧 OMP 路径"每线程构造一份 mt19937"会让两次调用产生相同序列，新实现单线程 + mutex 推进确保不会再退化。

### 1.3 状态恢复（赛题 §三(3)）

赛题硬性要求："保存 state → 继续生成 → 恢复 state → 再次生成 → 验证恢复后结果与原序列对齐"。CPU 必须覆盖；CUDA 若实现也应覆盖。

| 验收点 | 测试用例 | 文件:行 |
|---|---|---|
| state blob 往返保 InitialSeed | `GeneratorStateTest.GetSetStateRoundtrip`（CPU+CUDA） | `tests/generator/test_state.cc:33` |
| **完整循环：draw → save → draw₁ → restore → draw₂，验 draw₁==draw₂** | `GeneratorStateTest.GetSetStateRestoresSequence`（CPU+CUDA） | `tests/generator/test_state.cc:49` |
| CPU state 格式校验 4 case | `CpuStateValidation.{TruncatedHeader, BadMagic, BadVersion, PayloadSizeMismatch}` | `tests/generator/cpu_only/test_state_validation.cc:27-48` |
| CUDA state 格式校验 + 跨格式拒绝 | `CudaStateValidation.{RoundtripPreservesSeedOffsetInitialSeed, TruncatedHeader, BadMagic, BadVersion, PayloadSizeMismatch, RejectsCpuStateBlob}` | `tests/generator/cuda_only/test_cuda_state_validation.cc:29-78` |

**全部通过。**`RejectsCpuStateBlob` 实现"`SetState()` 校验来源是否同类型 Generator"——把 CPU magic 的 blob 喂给 `CUDAGeneratorImpl::SetState` 抛 `std::runtime_error`。

### 1.4 默认 Generator 行为（赛题 §三(4)）

| 验收点 | 测试用例 | 文件:行 |
|---|---|---|
| 默认 Generator 多次取得同一来源 | `GeneratorDefaultTest.DefaultGeneratorIsStable` | `tests/generator/test_default.cc:15` |
| `manual_seed` 影响默认 Generator | `GeneratorDefaultTest.ManualSeedTouchesDefault` | `:22` |
| **首次访问前 manual_seed 也生效**（`last_user_seed_` 记忆） | `GeneratorDefaultTest.ManualSeedBeforeFirstAccessRemembersSeed` | `:28` |
| 不同 CUDA 设备默认 Generator 独立 | `GeneratorDefaultMultiGpu.Cuda0AndCuda1HaveIndependentDefaults`（多 GPU 主机才 RUN，1-GPU SKIP） | `tests/generator/cuda_only/test_default_multi_gpu.cc:27` |
| 不传 generator 时 fallback 到默认 | `GeneratorDispatchTest.NullGeneratorFallsBackToDefault` | `tests/generator/test_dispatch.cc:18` |
| 显式传入 Generator 时不会误用默认 | 同上 fixture（验证显式与默认序列不同） | 同 |

**全部通过**（多 GPU SKIP 在 1-GPU 主机上属预期）。

### 1.5 主流框架语义对齐（赛题 §三(5)）

参考 PyTorch `c10::Generator` / `at::CPUGeneratorImpl` / `at::CUDAGeneratorImpl` / Randomness Notes（链接见 §2.1）。

| 对齐场景 | 验证手段 | 文件:行 / 测试名 |
|---|---|---|
| 手动 seed 后结果可复现 | `GeneratorSeedTest.ManualSeedReseedsState` + 各算子 `SameSeed*` | （见 §1.2） |
| Dropout 同 seed 可复现 | `GeneratorDropoutOpTest.SameSeedSameMask` | `tests/generator/test_ops_dropout.cc:49` |
| Dropout 反向 mask·dy/(1-p) | `GeneratorDropoutOpTest.BackwardEqualsMaskScale` | `:104` |
| `Module::Eval()` 关闭 Dropout 随机性 | `GeneratorDropoutOpTest.EvalGateBypassesDropout` + `nn::function::Dropout(training=false)` 旁路 | `:121` / `:129` |
| `Module::Train/Eval` 递归切换 | `ModuleTrainingTest.{DefaultIsTrainingTrue, EvalRecursesIntoChildren, RecursesIntoPpPrefixedChildren, NullChildSkipped}` | `tests/generator/test_module_training.cc:32-72` |
| 随机张量生成支持显式 + 默认 Generator | `nn::function::Rand/Randn`（`nn/functional.cc:91/101`）走 `ResolveGenerator` 通过 dispatcher 调 `UniformRandom/NormalRandom`；与 `torch.rand/randn` 接口语义一致 | — |
| state 保存恢复语义 | §1.3 完整循环测试 + state-validation 系列 | （见 §1.3） |

**全部通过**。逐元素 bit 一致**未要求**也**未实现**（赛题 §三 末段明确不要求）；CPU 与 CUDA 同 seed 不一致**未要求**也**不可能**（引擎不同），原因详见 §2.4。

### 1.6 整体测试结果

Train-server (CUDA Debug, USE_CUDA=ON USE_NCCL=ON) HEAD `1307669` 完整 ctest：

```
98% tests passed, 10 tests failed out of 532
(534 total - 2 Disabled = 532 ran; 10 fail / 12 skip / 510 pass)
Total Test time (real) = 90.61 sec
```

Generator suite 单跑（`-R 'Generator|ModuleTrainingTest'`）：

```
100% tests passed, 0 tests failed out of 68
(34 CPU + 34 CUDA; 1 multi-GPU SKIP on this 1-GPU host)
```

10 fail 是 pre-existing 上游 `gemm.cu:41 Check failed: p.stride_a == 0LL` DCHECK（7 个 Matmul + 3 个 LoRA），**与本 PR 完全无关**：已在 pristine upstream master `a917760` 单独 build 复现同一组 10 个测试名失败（详见 §3.5）。12 skip 是 1 个多 GPU 测试 + 1 个 TensorCopy 多设备 + 10 个 CPU `TransformerModuleTest`（基础设施限制）。

复现脚本 `scripts/check_reproducibility.sh build` 在 train-server CUDA build 上：

```
PASS: 68 test outcomes match across two ctest runs
```

CPU-only 本地 build：`PASS: 34 test outcomes match across two ctest runs`，exit 0。

---

## 2. 对齐性与行为分析报告

### 2.1 设计参考与命名映射

参考 PyTorch 文档与源码：[`torch.Generator`](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#generator)、[`Generator.h`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/Generator.h)、[`CPUGeneratorImpl.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/CPUGeneratorImpl.cpp)、[`CUDAGeneratorImpl.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp)、[Randomness Notes](https://docs.pytorch.org/docs/stable/notes/randomness.html#pytorch-random-number-generator)。

| 本实现 | PyTorch 对应 | 异同 |
|---|---|---|
| `Generator` 句柄（PImpl，`shared_ptr<GeneratorImpl>`） | `c10::Generator`（`intrusive_ptr<GeneratorImpl>`） | 浅拷贝、设备绑定语义一致；引用计数实现差异 |
| `core::GeneratorImpl` 抽象基类 | `c10::GeneratorImpl` | mutex + device + 抽象 seed/state 接口；磁盘格式不同（本实现用 byte vector + magic header；PyTorch 用 Tensor） |
| `CPUGeneratorImpl`（`std::mt19937_64`） | `at::CPUGeneratorImpl`（mt19937 + 自有状态布局） | 引擎不同 → bit 不一致，语义一致 |
| `CUDAGeneratorImpl`（Philox4_32 + offset） | `at::CUDAGeneratorImpl`（Philox4_32 + offset） | **vendored 同一份 `philox.cuh`**（BSD-3 attribution 保留）；调用约定一致；只服务于 random kernel，不接 distribution 模板 |
| `default_generator(Device)` | `at::detail::DefaultGenerator` 子系统 | 入口形态与 `at::globalContext().defaultGenerator(...)` 语义一致 |
| `manual_seed` / `manual_seed_cuda` | `torch.manual_seed` / `torch.cuda.manual_seed_all` | 入口名不同（snake_case vs `torch.*`），效果一致：CPU + 全部 CUDA 默认同步刷新 |
| `nn::Module::Train(bool)/Eval()/IsTraining()` | `Module.train()/eval()/training` | 递归到所有子模块；本实现**不跳过 `__pp_*`**（说明见 §2.4 之 5） |
| `nn::Dropout` / `nn::function::Dropout` | `nn.Dropout` / `nn.functional.dropout` | mask `kUINT8` vs PyTorch `bool`（语义等价）；缺 `inplace` 参数（见 §2.4 之 4） |
| `nn::function::Rand/Randn` | `torch.rand/randn` | dtype 仅 FP32（见 §2.4 之 3） |

### 2.2 接口语义对照

#### 2.2.1 句柄（`infini_train/include/generator.h`）

| 本实现 | PyTorch | 备注 |
|---|---|---|
| `Generator(Device)` (L18) | `Generator(device=...)` | |
| `ManualSeed(uint64_t)` (L25) | `manual_seed(seed)` | 命名差异 |
| `Seed()` / `InitialSeed()` (L26-27) | `seed()` / `initial_seed()` | |
| `GetState() -> vector<uint8_t>` (L29) | `get_state() -> Tensor` | 返回类型差异（见 §2.4 之 1） |
| `SetState(span<const uint8_t>)` (L30) | `set_state(Tensor)` | 同上 |
| `device()` (L32) | `gen.device` | |
| 浅拷贝（`shared_ptr` 共享 Impl） | 浅拷贝（`intrusive_ptr` 共享） | 实现不同，语义一致 |

#### 2.2.2 基类（`infini_train/include/core/generator/generator_impl.h`）

`mutex_` + `device_` + `StateMagic()` + 抽象 `SetCurrentSeed/CurrentSeed/InitialSeed/GetState/SetState/Reseed`（L25-38），对应 `c10::GeneratorImpl`。基类 `Seed()` 实现（`generator_impl.cc:27`）调子类 `Reseed` 但**不改 initial_seed_**，与 PyTorch `Generator::seed()` 语义一致。

#### 2.2.3 默认池 + 全局 seed

- `default_generator(Device)`：`generator.h:46` + `generator_impl.cc:181`，多次取共享同一句柄（`std::once_flag`，L167-194）。
- `manual_seed(uint64_t)` → `ResetAllSeeds`（L96）：CPU 默认 reseed + 已初始化 CUDA 默认 reseed + 把 seed 记到 `last_user_seed_[device]`，让"先 manual_seed 再首访"也能拿到该 seed（`LastUserSeedOrRandom`，L61-69）。对应 `torch.manual_seed`。
- `manual_seed_cuda(uint64_t)` → `ResetCudaSeeds`（L116）：仅刷 CUDA。对应 `torch.cuda.manual_seed_all`。

#### 2.2.4 Module training（`infini_train/include/nn/modules/module.h`）

- `Train(bool)/Eval()/IsTraining()`（L92-94）↔ `Module.train()/eval()/training`。
- `module.cc:288-296` 递归 `modules_`，仅跳 `nullptr`；**对 `__pp_*` 前缀不跳**（说明见 §2.4 之 5）。

#### 2.2.5 Dropout

- `autograd::Dropout`（`autograd/dropout.cc:22-27`，`saved_tensors_={mask}`）↔ `aten::native_dropout`（多返回）。
- `nn::Dropout`（`nn/modules/dropout.h:16`，`IsTraining()` 门控）↔ `nn.Dropout`，**缺 `inplace`**（§2.4 之 4）。
- `nn::function::Dropout(input, p, training, optional<Generator>)`（`functional.cc:111`）↔ `nn.functional.dropout`。
- mask dtype `kUINT8`（`cpu/dropout.cc:30`、`cuda/dropout.cu:30`）↔ PyTorch `bool`，语义等价。

#### 2.2.6 随机张量生成 + 初始化

- `nn::function::Rand/Randn`（`functional.cc:91/101`）↔ `torch.rand/randn`，FP32 only（§2.4 之 3）。
- `nn::init::Uniform/Normal/KaimingUniform`（`init.cc:91/21/78`）↔ `nn.init.uniform_/normal_/kaiming_uniform_`，全部走 dispatcher、统一接收 `std::optional<Generator>`，未传则 `ResolveGenerator` 取设备默认。

### 2.3 行为已对齐项

> 全部通过。每条对齐均由文件:行（实现侧） + 测试名（验证侧）双重佐证。

1. **同 seed 复现**：`generator_impl.cc:81`（`ManualSeed`）→ 各 ImplmpL `SetCurrentSeed`；测试见 §1.2 的 `*SameSeed*` 系列。
2. **状态推进 trap（连续两次同长序列必须不同）**：`uniform_random.cc:22`（CPU 单线程 + mutex）、`cuda/uniform_random.cu:50-58`（Philox `subsequence=tid, offset=offset_4tuple*4`，调完用 `NextPhiloxState(n)` 推进 offset）；测试 `GeneratorUniformOpTest.ConsecutiveCallsAdvanceState`。
3. **State 完整循环对齐**（赛题 §三(3) 硬性要求）：`GeneratorStateTest.GetSetStateRestoresSequence`（CPU+CUDA 双实例化）。
4. **InitialSeed 不被 Seed() 覆盖**：`generator_impl.cc:27-32` + `cpu_generator_impl.cc:44-47` + `cuda_generator_impl.cu:42-49`；测试 `GeneratorInitialSeedTest.SeedDoesNotChangeInitialSeed`。
5. **默认池一致性**：`generator_impl.cc:167-194`（`std::once_flag`）；测试 `GeneratorDefaultTest.DefaultGeneratorIsStable`，多 GPU 独立性 `GeneratorDefaultMultiGpu.Cuda0AndCuda1HaveIndependentDefaults`。
6. **显式 Generator vs 默认 fallback**：`ResolveGenerator`（`generator_impl.cc:200-205`）被 `init.cc:25`、`functional.cc:96/106/116`、`autograd/dropout.cc:20` 调用；测试 `GeneratorDispatchTest.NullGeneratorFallsBackToDefault`。
7. **Dropout 行为**：`out_i = mask_i ? x_i/(1-p) : 0`、`dx_i = mask_i ? dy_i/(1-p) : 0`，`mask_i ∈ {0,1}` 存于 `kUINT8`；CPU `cpu/dropout.cc:51-58/77-79`，CUDA `cuda/dropout.cu:30-36/40-44`；6 case 见 §1.2。
8. **`Module.Train/Eval` 递归**：`module.cc:288-296`；4 case 见 §1.5。
9. **跨格式 state 拒绝**：`CUDAGeneratorImpl::SetState` 校验 magic，喂入 `'CPUG'` blob 抛 `std::runtime_error`；测试 `CudaStateValidation.RejectsCpuStateBlob`。

### 2.4 行为未完全对齐项及原因

> 5 项已知未对齐，原因均已在 spec § 中声明，不属于回归。

1. **同 seed 与 PyTorch bit 不一致**：CPU mt19937_64 与 PyTorch mt19937 引擎参数不同；CUDA Philox 虽 vendored 同一份 `philox.cuh`，但 `subsequence/offset` 调用约定与 PyTorch dispatch 路径不同；赛题 §三 末段与 spec §2.2 均明确不要求 bit 一致。
2. **CPU 与 CUDA 同 seed 不一致**：引擎不同必然不同，赛题 §二(5) 明确"不要求 CPU 与 CUDA 在相同 seed 下生成逐元素一致的随机结果"，spec §6 显式声明。
3. **dtype 仅 FP32**：随机 kernel `CHECK_EQ(dtype, kFLOAT32)` fail-fast（`cpu/uniform_random.cc:18`、`cuda/uniform_random.cu:40`、`cpu/dropout.cc:22`、`cuda/dropout.cu:53`、`cpu/normal_random.cc:18`、`cuda/normal_random.cu:38`）。BF16/FP16/FP64 不在本期范围；解除路径见 §2.5 之 2。
4. **`nn::Dropout` 缺 `inplace` 参数**：`nn/modules/dropout.h:20` 仅接 `(p, generator)`，没有 `inplace=False/True` 旋钮；语义等价（默认就是 out-of-place），多一份显存 buffer；解除路径需要 autograd 侧支持原位写。
5. **`Module.Train/Eval` 不跳过 `__pp_*` 前缀**：`StateDict()` 跳 `__pp_*` 是序列化去重需求（pipeline-parallel chunk 内的 layer 别名暴露主名给 checkpoint）；`Train()` 是幂等 setter 无去重需求；如果跳过则未来 PP 改别名策略时 `chunk._pp_layer.dropout` 会被静默漏切到 eval。**这与赛题无 hard 冲突**，但与 PyTorch `Module.train()` 默认递归到所有子模块的行为一致；选择不跳是更安全的语义。
6. **OMP 多线程 RNG 未优化**：现 CPU 路径单线程 + mutex；spec §2.2 明确"非目标"。解除路径见 §2.5 之 3。

### 2.5 当前实现范围与可扩展方向

1. LoRA `LoRAConfig::dropout` 接线（spec §5.5、Phase 3 §1）：当前 LoRA 模块未消费 dropout 字段；接 `nn::Dropout` 即可。
2. **BF16/FP16 dtype**：解除 random kernel 的 `CHECK_EQ`，按 dtype 模板分发（参考 `Cast`/`Fill` 实现），CUDA 侧可复用 Philox 4-tuple → 4× half conversion。
3. **OMP 多线程 RNG**：每线程独立 sub-stream + Philox `jump-ahead`；CPU mt19937_64 也可换成 Threefry/Philox 让 CPU/CUDA 引擎对齐（同时也让 §2.4 之 2 的不一致变得可选解除）。
4. **CUDA graph capture 友好 Philox state snapshot**（Phase 3 spec §2.6）：当前 offset 在 host 侧 mutex 推进、kernel launch 时 capture，graph capture 路径需要把 offset 作为 graph external input。
5. **Bernoulli / randint / categorical**：复用 `ResolveGenerator` + dispatcher，添 kernel 注册即可。
6. **ZeRO/分布式 RNG**（spec §10）：per-rank derived seed `seed + rank`，结合数据并行 / pipeline 并行的 micro-batch shape 决定 sub-stream 划分。

---

## 3. 测试与可复现性说明

### 3.1 编译环境

CUDA Debug 全量构建：

```bash
PATH=/usr/local/cuda/bin:$PATH cmake -B build \
  -DBUILD_TEST=ON -DUSE_CUDA=ON -DUSE_NCCL=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j 4   # -j 不限会触发 cc1plus OOM, 详见 PR-prep §10
ctest --test-dir build --output-on-failure
```

CPU-only（本地无 GPU 时）：

```bash
cmake -B build_cpu -DBUILD_TEST=ON -DUSE_CUDA=OFF -DUSE_NCCL=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5  # 兼容 modern CMake (≥4.x) 的 gflags submodule
cmake --build build_cpu -j
ctest --test-dir build_cpu --output-on-failure
```

`-DBUILD_TEST=ON` 是必须的（`CMakeLists.txt:7` 的 `option(BUILD_TEST OFF)`）。若 train-server 这种受限环境，注意：nvcc 不在非交互 shell 的 PATH，必须显式设；`third_party/eigen/` working tree 缺失时需手动恢复（rsync 或 `git submodule update --init third_party/eigen`）。

### 3.2 完整 ctest 期望结果

Train-server (CUDA Debug) HEAD `1307669`：

```
98% tests passed, 10 tests failed out of 532
(534 - 2 Disabled = 532 ran; 510 pass / 10 fail / 12 skip)
Total Test time (real) = 90.61 sec
```

10 fail 全部是 pre-existing 上游 `gemm.cu:41 Check failed: p.stride_a == 0LL` DCHECK abort（属性归因详见 §3.5）。12 skip 见 §1.6。

### 3.3 Generator 子集 ctest

```bash
ctest --test-dir build --output-on-failure -R 'Generator|ModuleTrainingTest'
```

期望（train-server CUDA build）：

```
100% tests passed, 0 tests failed out of 68
(34 CPU + 34 CUDA; 1 SKIP for GeneratorDefaultMultiGpu on 1-GPU host)
```

**注意 filter 形式：**赛题报告的 ctest 名是 gtest fixture 风格 `CPU/GeneratorBasicTest.*`、`CUDA/GeneratorStateTest.*` 等，**不是** `test_generator_*` 那种 binary 名。早期 `scripts/check_reproducibility.sh` 写错过 filter（提交 `9de4b0e` 用 `^test_generator_`），fix 在 `edc5265`。

### 3.4 加分项：自动复现脚本 `scripts/check_reproducibility.sh`

```bash
./scripts/check_reproducibility.sh build       # CUDA build
./scripts/check_reproducibility.sh build_cpu   # CPU-only build
```

实现：连续跑两次 `ctest -R 'Generator|ModuleTrainingTest'`，逐测试提取 `Test +#[0-9]+: <name> ... <Passed|Failed|Skipped>` verdict 行（grep 后 sed 砍掉时间字段），diff 两次结果，全部 match 则 exit 0 + `PASS: N test outcomes match across two ctest runs`。空输出（stale build）走 fail-fast `ERROR: no Generator test verdicts captured` exit 3，避免 pipefail 静默吞错。

期望输出：

- Train-server CUDA build：`PASS: 68 test outcomes match across two ctest runs`，exit 0。
- 本地 CPU-only build：`PASS: 34 test outcomes match across two ctest runs`，exit 0。

### 3.5 上游 `gemm.cu:41` 失败的实证归因

10 fail 测试名（train-server feat/generator-phase1）：

```
261 - CUDA/AutogradForwardTest.MatmulForward
330 - CUDA/AutogradMatmulBackwardTest.MatmulBackward
331 - CUDA/AutogradMatmulBackwardTest.MatmulBackwardSquare
332 - CUDA/AutogradMatmulBackwardTest.MatmulBackwardDifferentShapes
333 - CUDA/AutogradMatmulForwardTest.MatmulForward
334 - CUDA/AutogradMatmulForwardTest.MatmulDifferentShapes
336 - CUDA/AutogradMatmulForwardTest.MatmulSquare
383 - CUDA/LoRATest.LoRALinearMerge
389 - CUDA/LoRATest.GetLoRAModel
390 - CUDA/LoRATest.MergeAndUnload
```

均 abort 于 `gemm.cu:41] Check failed: p.stride_a == 0LL (... vs. 0)`。

**静态归因（git）：**

- `git blame infini_train/src/kernels/cuda/common/gemm.cu` → DCHECK 由 `5dfd4b2e`（chen, 2026-04-15）引入，`fd7c3a1c`（chen, 2026-05-07）refactor，**两个 commit 都在 `origin/master`**、远早于本分支起点 2026-05-21。
- `git diff master..HEAD --stat` → 本 PR 不触碰 `gemm.cu`、`matmul*`、`lora*` 任何文件。
- `git log master..HEAD -- 'gemm.cu' '*matmul*' '*lora*'` → 空。

**实证归因（在 train-server 上单独 build pristine upstream master `a917760` 跑 ctest）：**

```
98% tests passed, 10 tests failed out of 403
Total Test time (real) = 78.86 sec
```

10 fail：

```
210 - CUDA/AutogradForwardTest.MatmulForward
279 - CUDA/AutogradMatmulBackwardTest.MatmulBackward
280 - CUDA/AutogradMatmulBackwardTest.MatmulBackwardSquare
281 - CUDA/AutogradMatmulBackwardTest.MatmulBackwardDifferentShapes
282 - CUDA/AutogradMatmulForwardTest.MatmulForward
283 - CUDA/AutogradMatmulForwardTest.MatmulDifferentShapes
285 - CUDA/AutogradMatmulForwardTest.MatmulSquare
332 - CUDA/LoRATest.LoRALinearMerge
338 - CUDA/LoRATest.GetLoRAModel
339 - CUDA/LoRATest.MergeAndUnload
```

测试编号不同（master 无新 Generator suite 所以总数少 129），**测试名与失败签名 1:1 完全相同**：7 Matmul + 3 LoRA，全部 `gemm.cu:41 stride_a==0` DCHECK。

**结论：上游 `master@a917760` 已带这 10 个失败；本 PR 与之无关。**详见 [PR-prep notes §10](superpowers/notes/2026-05-21-phase2-pr-prep.md#10-empirical-proof-pristine-upstream-master-has-the-same-10-failures-2026-05-21)。

---

## 4. 验收清单 — 赛题 §五 逐条命中

### 验收要求（必需）

- [x] 代码以 PR 形式提交 — `guozhihao-224:feat/generator-phase1` → `InfiniTensor:master`，37 commits 分阶段（Phase 1 抽象+CPU、Phase 2 CUDA、Phase 3 Dropout/Module training、Phase 4 报告 + 复现脚本 + 实证）。
- [x] 完成统一 Generator 抽象 — `Generator` PImpl + `GeneratorImpl` 基类 + Registry。详见 §2.2。
- [x] 实现 CPU Generator — `CPUGeneratorImpl`（mt19937_64）。详见 §0、§2.1。
- [x] CUDA Generator（建议项已实现） — `CUDAGeneratorImpl`（Philox4_32 + offset）。详见 §0、§2.1。
- [x] 默认 Generator 管理机制 — CPU 单例 + 每 CUDA 设备独立。详见 §1.4、§2.2.3。
- [x] 统一全局随机种子入口 — `manual_seed` / `manual_seed_cuda`。详见 §2.2.3。
- [x] 默认 + 显式 Generator 两条路径 — `ResolveGenerator`。详见 §1.1、§2.3 之 6。
- [x] State 获取与恢复 — `GetState/SetState` + magic/version header；完整循环测试 §1.3。
- [x] 至少改造一类初始化 + 一类训练期随机算子 — 初始化：Uniform/Normal/KaimingUniform/Tensor::Uniform；训练期：Dropout + Rand/Randn。详见 §0。
- [x] 测试或日志验证可复现性 — 68 个 Generator suite 测试 + `scripts/check_reproducibility.sh`。详见 §3.3、§3.4。

### 加分项

- [x] 设计清晰、接口与实现分层合理 — PImpl 句柄 + Impl 基类 + Registry，外部接口不暴露 mt19937/curand/Philox 细节（全部在 `infini_train/include/core/generator/*` 内）。
- [x] 测试覆盖充分 — 68 case，包含 seed/state/默认/显式/跨设备/格式校验/state 完整循环/多 GPU 独立性/state-advance trap。
- [x] 调用处改造完全 — `nn::init::*` 全部走 dispatcher；`example/` 中旧 `static std::mt19937 gen` 与 `kRandomSeed=42` 全部清除（commit `dc26345`）。
- [x] 与 PyTorch 接口语义和行为分析完整 — §2.1 命名映射 + §2.2 接口 6 个子表 + §2.3 9 项已对齐 + §2.4 6 项未对齐 + §2.5 6 项扩展方向。
- [x] 加分项 `scripts/check_reproducibility.sh` — §3.4，本地 CPU + train-server CUDA 两侧均 exit 0。
- [ ] PR 经过完整 review 流程，达到可合入标准 — 本分支已 push 到 `guozhihao-224/feat/generator-phase1`，待向 `InfiniTensor/master` 开 PR 后由上游 reviewer 走 review 流程；本仓库内部已经 subagent-driven-development（implementer → spec-compliance reviewer → code-quality reviewer → controller）二阶段 review，每个 Phase 2/3/4 任务都通过。

---

## 附录 A：相关文件索引

| 类别 | 文件 |
|---|---|
| 抽象与默认池 | `infini_train/include/generator.h`, `infini_train/include/core/generator/{generator_impl.h, cpu_generator_impl.h, cuda_generator_impl.h}` |
| 抽象与默认池实现 | `infini_train/src/core/generator/{generator_impl.cc, cpu_generator_impl.cc, cuda_generator_impl.cu, philox.cuh}` |
| CPU 随机 kernel | `infini_train/src/kernels/cpu/{uniform_random.cc, normal_random.cc, dropout.cc}` |
| CUDA 随机 kernel | `infini_train/src/kernels/cuda/{uniform_random.cu, normal_random.cu, dropout.cu}` |
| 算子接入 | `infini_train/src/nn/{init.cc, functional.cc, modules/dropout.{h,cc}, module.cc}` |
| autograd Dropout | `infini_train/src/autograd/dropout.{h,cc}` |
| 测试 | `tests/generator/`（参数化 + `cpu_only/` + `cuda_only/`） |
| 复现脚本 | `scripts/check_reproducibility.sh` |
| 实测留档 | `docs/superpowers/notes/2026-05-21-phase2-pr-prep.md`（§8 PR body 模板、§9 Phase 3+4 addendum、§10 master 实证） |
| 设计 spec | `docs/superpowers/specs/2026-05-19-generator-abstraction-design.md`、`2026-05-21-generator-phase3-dropout-design.md` |
