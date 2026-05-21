# InfiniTrain Generator 抽象 — 技术报告

> **受众 / 目的**：(1) 让赛题 reviewer 在 5 分钟内确认 §五 验收命中、§四 报告齐备；(2) 给未来维护者留下关键设计决策与待办。
>
> **范围**：分支 `feat/generator-phase1`，HEAD `7f91e6c`，自 `master` 起 38 commits。已 push 到 `guozhihao-224/feat/generator-phase1`。
>
> 详细实测留档与 PR body 草稿：[`docs/superpowers/notes/2026-05-21-phase2-pr-prep.md`](superpowers/notes/2026-05-21-phase2-pr-prep.md)。

---

## TL;DR

- **做了什么**：在 InfiniTrain 引入 PyTorch `c10::Generator` 等价的 PImpl 句柄 + 多态 Impl + 设备默认池 + 全局 seed 入口；CPU 后端 `mt19937_64` + CUDA 后端 vendored Philox4_32（BSD-3）；改造 `nn::init::{Uniform,Normal,KaimingUniform}`、`nn::Dropout`、`nn::function::{Rand,Randn}`；新增 `nn::Module::Train/Eval` 训练态切换。
- **结果**：新增 68 个 Generator 测试 100% 通过；复现脚本双跑 verdict 全 match；完整 ctest **510 pass / 10 fail / 12 skip / 532 ran**。10 fail 经实证为上游 `gemm.cu:41 stride_a` DCHECK——pristine `master@a917760` 单独 build 同样 393/10/403 失败相同 10 个测试名（详见 §6.4）。
- **状态**：赛题 §五 必需项 **10/10** ✅、加分项 **5/6** ✅（最后 1 项"PR review 流程"待 PR 开出后由上游推进）。

---

## 1. 架构与实现原理

### 1.1 架构

```
              user code
                  │  显式 Generator | 不传
                  ▼
          ┌──────────────┐
          │  Generator   │   PImpl 句柄（shared_ptr<GeneratorImpl>），ManualSeed / Seed
          └──────┬───────┘   InitialSeed / GetState / SetState / device()
                 │
                 ▼
          ┌──────────────┐
          │ GeneratorImpl│   抽象基类：mutex_、device_、StateMagic()
          └──┬────────┬──┘   纯虚 SetCurrentSeed/CurrentSeed/Reseed/GetState/SetState
             │        │
             ▼        ▼
   CPUGeneratorImpl  CUDAGeneratorImpl
   mt19937_64        Philox4_32(vendored)，16B 状态 (seed,offset,initial_seed)
   magic 'CPUG'      magic 'CUDG'

  default_generator(Device)  per-Device 单例（std::once_flag 惰性初始化）
  manual_seed(uint64_t)      全局入口：CPU + 全部 CUDA 默认同步刷新
  ResolveGenerator(opt,Dev)  op 入口：null → 默认池；非 null → 显式
        │
        ▼
  Dispatcher::Call<R>({device.type(), "OpName"}, …, impl)
        →  UniformRandom / NormalRandom / DropoutForward / DropoutBackward
```

**封装契约**：`Generator` 头文件不出现 `mt19937`/`curand`/`Philox` 任何一个类型；后端切换不破坏调用方。

### 1.2 实现原理

**(a) 状态机与序列化（基类，所有后端共享）**
`GeneratorImpl` 持 `mutex_`、`device_`、`current_seed_`、`initial_seed_`。`ManualSeed(s)` 同时刷 current+initial；`Seed()` 仅刷 current（PyTorch 语义）。`GetState()/SetState()` 在子类实现，统一头部 `magic(4) | version(4) | payload_size(8) | payload`：`SetState` 校验 magic 匹配本后端 + version=1 + payload 长度匹配，三者任一不通过抛 `runtime_error`——这让把 CPU state blob 喂给 CUDA Impl 显式失败而非 silently corrupt（`CudaStateValidation.RejectsCpuStateBlob`）。

**(b) CPU 后端**
`std::mt19937_64`（注意是 64-bit 变体，与 PyTorch 32-bit `mt19937` 不同——bit 不一致 by design）。`UniformRandom` kernel 在 mutex 下用 `std::uniform_real_distribution<float>(lo,hi)` 单线程填张量；`NormalRandom` 用 `std::normal_distribution<float>(μ,σ)`。**不走 OMP 是有意的**：旧路径"每线程独立构造 mt19937"会让两次同长度连续调用产出相同序列（trap test `ConsecutiveCallsAdvanceState` 专门拦这个），单线程 + mutex 强制状态推进可见。

**(c) CUDA 后端**
vendored PyTorch Philox4_32 device engine（BSD-3 attribution 保留在 `philox.cuh` 顶部）。`CUDAGeneratorImpl` 持 16-byte 状态 `{seed, offset, initial_seed}`，offset 单位是 4-tuple（每次 Philox 一发产 4 个 uint32）。

随机 kernel 的 launch 三步走（**全在 mutex 下**）：

1. 读 `offset_pre = NextPhiloxState(n)`——把 offset 推进 `ceil(n/4)` 个 4-tuple，返回推进*前*的值；
2. 用 `(seed, subsequence=tid, offset=offset_pre*4)` 在 device 上构造 `philox4x32`，每 thread 拿到独立无碰撞的子序列；
3. 通过 `dynamic_cast<CudaStream*>(...)->cuda_stream()` 在当前 device 的 stream 上 launch。

NormalRandom 走 Box-Muller，每 thread 一次产 2 值，`NextPhiloxState(2*n)` 是 upper bound 保证 offset 推进 ≥ 实际消费。Dropout p==0 走 `cudaMemcpyAsync + cudaMemsetAsync` 快路径，不动 RNG offset。

**(d) 默认池 + 全局 seed**
`default_generator(Device)` 用 per-device `std::once_flag` 惰性构造（CPU 1 个 + 每 CUDA device 1 个）。`manual_seed(s)` 同时刷新所有已构造的默认 Impl，并把 `s` 缓存进 `last_user_seed_`——这样"先 `manual_seed` 再首次 `default_generator(cuda:1)`"也能让晚构造的默认 Impl 用上 user seed（`ManualSeedBeforeFirstAccessRemembersSeed`）。`manual_seed_cuda` 仅刷 CUDA 默认。

**(e) 算子接入**
op 入口统一调 `ResolveGenerator(std::optional<Generator>, Device)`：传了就用，没传则取 `default_generator(device)`，永远返回非空 Impl 指针。然后 `Dispatcher::Instance().Call<R>({device.type(), "OpName"}, …, impl)` 让 dispatcher 路由到 CPU 或 CUDA kernel。`DropoutForward` 返回 `tuple<output, mask>`，所以模板参数显式写 `Dispatcher::Call<std::tuple<...>>`。

**(f) 训练态切换**
`nn::Module::Train(bool)/Eval()/IsTraining()` 递归 `modules_`（包括 `__pp_*` 别名子模块，仅跳 `nullptr`——见决策表 #4）。`nn::Dropout` 不持私有 `training_`，由 `IsTraining()` 直接门控；eval 态时 `nn::Dropout::Forward` 直接返回 input，不走 RNG。

### 1.3 关键决策（含替代方案）

| # | 决策 | 替代方案 | 选择理由 |
|---|---|---|---|
| 1 | CUDA 引擎选 vendored PyTorch Philox4_32 | curandStateXORWOW / curandStatePhilox4_32_10 / 自实现 PCG | 与 PyTorch 引擎家族对齐，方便后续 Bernoulli/randint 复用；offset 推进可控；BSD-3 attribution 已保留 |
| 2 | State blob 格式：`magic(4)\|version(4)\|payload_size(8)\|payload` | PyTorch 的 Tensor wrapper；`std::string` | InfiniTrain 暂无成熟 byte-tensor 序列化路径；自定 header 让 cross-device 校验显式抛 `runtime_error`（`CudaStateValidation.RejectsCpuStateBlob`） |
| 3 | Dropout mask 用 `kUINT8` | `kBOOL` / `float` mask | InfiniTrain 没有 `kBOOL` dtype；`float` 浪费 4× 显存；`kUINT8` 与 PyTorch `bool` 语义等价（仅取 0/1） |
| 4 | `Module::Train(bool)` 递归**不**跳 `__pp_*` | 与 `StateDict()` 对齐跳过 | `StateDict()` 跳是为序列化去重（PP chunk 别名）；`Train()` 是幂等 setter 无去重需求；跳过会让未来 PP 改别名时静默漏切 chunk 内 Dropout 的 eval 开关 |
| 5 | CPU random kernel 单线程 + mutex | 每线程 sub-stream + Philox jump-ahead | 赛题 §2.2 明确性能非目标；正确性 trap (`ConsecutiveCallsAdvanceState`) 优先；后续可解（§7） |
| 6 | `nn::Dropout` 不带 `inplace` | 仿 `nn.Dropout(p, inplace=True)` | autograd 侧需要支持原位写，与现 `Function::Apply` 契约不兼容；out-of-place 多一份显存但语义同 PyTorch 默认 |

### 1.4 文件索引

| 模块 | 文件 |
|---|---|
| 抽象与默认池 | `infini_train/include/generator.h`, `infini_train/include/core/generator/{generator_impl,cpu_generator_impl,cuda_generator_impl}.h` |
| 实现 | `infini_train/src/core/generator/{generator_impl.cc, cpu_generator_impl.cc, cuda_generator_impl.cu, philox.cuh}` |
| CPU 随机 kernel | `infini_train/src/kernels/cpu/{uniform_random,normal_random,dropout}.cc` |
| CUDA 随机 kernel | `infini_train/src/kernels/cuda/{uniform_random,normal_random,dropout}.cu` |
| 算子接入 | `infini_train/src/nn/{init.cc, functional.cc, modules/dropout.{h,cc}, module.cc}` |
| autograd Dropout | `infini_train/src/autograd/dropout.{h,cc}` |
| 测试 | `tests/generator/`（参数化 + `cpu_only/` + `cuda_only/`） |
| 复现脚本 | `scripts/check_reproducibility.sh` |
| 设计 spec | `docs/superpowers/specs/2026-05-19-generator-abstraction-design.md`、`2026-05-21-generator-phase3-dropout-design.md` |

---

## 2. §五 验收一览

| # | 必需要求 | 命中 | 落点 |
|---|---|---|---|
| 1 | 代码以 PR 形式提交，结构清晰可 review | ✅ | 38 commits 分阶段；Phase 1/2/3/4 分别独立可 review |
| 2 | 统一 Generator 抽象 | ✅ | `Generator`(handle) + `GeneratorImpl`(base) + `GeneratorImplRegistry`，§1.1 |
| 3 | CPU Generator | ✅ | `CPUGeneratorImpl`(mt19937_64)，§1.2(b) |
| 4 | CUDA Generator（建议） | ✅ | `CUDAGeneratorImpl`(Philox4_32 + offset)，§1.2(c) |
| 5 | 默认 Generator 管理 | ✅ | CPU 单例 + 每 CUDA device 独立，`std::once_flag` 惰性初始化，§1.2(d) |
| 6 | 统一全局 seed 入口 | ✅ | `manual_seed` / `manual_seed_cuda` |
| 7 | 默认 + 显式两条路径 | ✅ | `ResolveGenerator(optional<Generator>, Device)`，§1.2(e) |
| 8 | State get/set | ✅ | byte blob + magic/version header + cross-magic reject，§1.2(a) |
| 9 | 改造 ≥1 初始化 + ≥1 训练期算子 | ✅ | 初始化：Uniform/Normal/Kaiming；训练期：Dropout + Rand/Randn |
| 10 | 提供测试或日志验证可复现性 | ✅ | 68 个 ctest case + `scripts/check_reproducibility.sh` |

| # | 加分项 | 命中 | 备注 |
|---|---|---|---|
| 1 | 设计清晰、分层合理 | ✅ | PImpl + Impl + Registry，外部不暴露 mt19937/curand/Philox |
| 2 | 测试覆盖充分 | ✅ | seed/state/默认/显式/跨设备/格式校验/state 完整循环/多 GPU/state-advance trap |
| 3 | 调用处改造完全 | ✅ | `nn::init::*` 全走 dispatcher；example 中 `static std::mt19937` + `kRandomSeed=42` 全部清除 |
| 4 | 与 PyTorch 接口语义和行为分析完整 | ✅ | §5 接口对照 9 项已对齐 + 5 项未对齐（含原因） + 6 项扩展 |
| 5 | 复现脚本（spec §9.1） | ✅ | `scripts/check_reproducibility.sh`，CPU + CUDA 双侧 exit 0 |
| 6 | PR 经过完整 review，达到可合入 | ⏳ | 仓内 subagent-driven review 全部通过（implementer → spec reviewer → quality reviewer → controller）；上游 review 待 PR 开出 |

---

## 3. 背景与约束

**背景**：InfiniTrain 当时缺统一 Generator 抽象、CPU/CUDA 双后端、默认池+全局 seed、随机算子接入、与 PyTorch 的语义一致性。本工作覆盖以上五点。参考 PyTorch [`torch.Generator`](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html) 与 ATen 的 [`Generator.h`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/Generator.h) / [`CPUGeneratorImpl.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/CPUGeneratorImpl.cpp) / [`CUDAGeneratorImpl.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp)。

**约束（非目标，由赛题 §三 末段或 spec §2.2 显式声明）**：

- 不要求与 PyTorch bit-for-bit 一致；
- 不要求 CPU 与 CUDA 同 seed 输出一致（引擎不同必然不同）；
- OMP 多线程 RNG 性能优化非目标；
- BF16/FP16/FP64 dtype 非本期范围。

---

## 4. 功能正确性验证（赛题 §四 之一）

**结论：68 个 Generator 测试 100% 通过；赛题 §三(1)-(5) 五类验收点全覆盖。** 测试通过 `INFINI_TRAIN_REGISTER_TEST` 自动 CPU+CUDA 双实例化（gtest 名 `CPU/Fixture.Case` 与 `CUDA/Fixture.Case`）。

下表把赛题 §三(1)-(5) 的每个具体要求映射到测试用例。完整 ctest 结果见 §6.2。

| 赛题要求（§三 子节） | 验证测试 | 文件:行 |
|---|---|---|
| **(1) 接口一致性** — seed 设置/获取、state 获取/恢复、设备查询、默认获取、显式 vs 默认 | `GeneratorBasicTest.{ConstructionAttachesDevice, ManualSeedRoundtripsCurrentAndInitial, CopyShareImpl}` | `tests/generator/test_generator_basic.cc:14-30` |
| | `GeneratorDispatchTest.NullGeneratorFallsBackToDefault` | `tests/generator/test_dispatch.cc:18` |
| | `CPUGeneratorImplTest.*` (5 case) / `CUDAGeneratorImplTest.*` (4 case) — 直接 Impl 接口 | `tests/generator/cpu_only/test_cpu_generator_impl.cc`, `cuda_only/test_cuda_generator_impl.cc` |
| **(2) 种子可复现** — 同 seed 一致、不同 seed 不同、状态推进 | `GeneratorSeedTest.{ManualSeedReseedsState, DifferentSeedsDifferState}` | `tests/generator/test_seed.cc:14/23` |
| | `GeneratorInitialSeedTest.SeedDoesNotChangeInitialSeed`（`Seed()` 不覆 InitialSeed） | `tests/generator/test_initial_seed.cc:12` |
| | `Generator{Uniform,Normal}OpTest.SameSeedSameOutput` + `GeneratorUniformOpTest.ConsecutiveCallsAdvanceState` (state-advance trap) | `tests/generator/test_ops_{uniform,normal}.cc` |
| | `GeneratorKaimingTest.SameSeedSameWeights`（参数初始化复现） | `tests/generator/test_ops_kaiming.cc:28` |
| | `GeneratorDropoutOpTest.{SameSeedSameMask, MaskKeepRateSane, OutputScaleCorrect}` | `tests/generator/test_ops_dropout.cc:49-86` |
| **(3) 状态恢复（赛题硬性）** — save → draw₁ → restore → draw₂，验 draw₁ ≡ draw₂ | `GeneratorStateTest.{GetSetStateRoundtrip, GetSetStateRestoresSequence}`（CPU+CUDA） | `tests/generator/test_state.cc:33/49` |
| | `Cpu/CudaStateValidation.*` (4+6 case) — 格式校验 + cross-magic reject | `tests/generator/{cpu,cuda}_only/test_*state_validation.cc` |
| **(4) 默认 Generator 行为** — 默认稳定、CPU/CUDA 自动选、显式不误用 | `GeneratorDefaultTest.{DefaultGeneratorIsStable, ManualSeedTouchesDefault, ManualSeedBeforeFirstAccessRemembersSeed}` | `tests/generator/test_default.cc:15-28` |
| | `GeneratorDefaultMultiGpu.Cuda0AndCuda1HaveIndependentDefaults`（多 GPU 独立；1-GPU host SKIP） | `tests/generator/cuda_only/test_default_multi_gpu.cc:27` |
| **(5) 主流框架语义对齐** — manual_seed 复现、Dropout 同 seed 一致、随机张量两条路径、state 语义 | 上面 (1)-(4) 全部 + `GeneratorDropoutOpTest.{BackwardEqualsMaskScale, EvalGateBypassesDropout, FunctionalTrainingFalseBypasses}` | `tests/generator/test_ops_dropout.cc:104-129` |
| | `ModuleTrainingTest.{DefaultIsTrainingTrue, EvalRecursesIntoChildren, RecursesIntoPpPrefixedChildren, NullChildSkipped}` | `tests/generator/test_module_training.cc:32-72` |

trap test 解释：`ConsecutiveCallsAdvanceState` 验"两次同长 Uniform 序列必须不同"。旧 OMP 路径"每线程独立构造 mt19937"会让两次调用产出相同序列；新实现单线程 + mutex 推进 + CUDA Philox 在 mutex 下推进 offset 后才 launch，确保不会再退化。

---

## 5. 与 PyTorch 对齐分析（赛题 §四 之二）

### 5.1 接口语义对照

| 维度 | 本实现 | PyTorch | 异同 |
|---|---|---|---|
| 句柄 | `Generator(Device)` | `c10::Generator(device=...)` | 浅拷贝、设备绑定一致；`shared_ptr` vs `intrusive_ptr` 实现差异 |
| 接口 | `ManualSeed/Seed/InitialSeed/GetState/SetState/device()` | `manual_seed/seed/initial_seed/get_state/set_state/device` | 一一对应；命名风格差异（CamelCase vs snake_case） |
| State 类型 | `vector<uint8_t>` | `Tensor` | InfiniTrain 缺 byte-tensor 序列化；语义等价 |
| CPU 引擎 | `mt19937_64` | `mt19937` | 不同变体 → bit 不一致（语义一致） |
| CUDA 引擎 | Philox4_32 (vendored) | Philox4_32 | 同份 `philox.cuh`；调用约定不同 |
| 默认池 | `default_generator(Device)` + `std::once_flag` | `at::detail::DefaultGenerator` | 形态一致 |
| 全局 seed | `manual_seed` / `manual_seed_cuda` | `torch.manual_seed` / `torch.cuda.manual_seed_all` | 行为一致（CPU + 全部 CUDA 同步刷新） |
| Module 训练态 | `Train(bool)/Eval()/IsTraining()` | `train()/eval()/training` | 递归到所有子模块；`__pp_*` 处理见决策 §1.3 之 4 |
| Dropout | `nn::Dropout(p, gen)` + `nn::function::Dropout(...)` | `nn.Dropout(p, inplace=False)` + `nn.functional.dropout` | 语义等价；缺 `inplace`（决策 §1.3 之 6） |
| Mask dtype | `kUINT8` | `bool` | 等价（仅 0/1） |
| 随机张量 | `nn::function::Rand/Randn` | `torch.rand/randn` | FP32-only |

### 5.2 已对齐项（实现 + 测试双重佐证）

| # | 行为 | 实现 | 验证 |
|---|---|---|---|
| 1 | 同 seed 复现 | `generator_impl.cc:81` | 各 `*SameSeed*` 测试 |
| 2 | 状态推进（trap） | `uniform_random.cc:22`（CPU mutex），`cuda/uniform_random.cu:50-58`（Philox `subsequence=tid, offset=offset_4tuple*4`，调完 `NextPhiloxState(n)`） | `ConsecutiveCallsAdvanceState` |
| 3 | State 完整循环 | byte-blob roundtrip + 实际序列对齐 | `GetSetStateRestoresSequence` |
| 4 | `Seed()` 不覆 `InitialSeed()` | `generator_impl.cc:27-32` | `SeedDoesNotChangeInitialSeed` |
| 5 | 默认池稳定 + 多 GPU 独立 | `generator_impl.cc:167-194`（`std::once_flag`） | `DefaultGeneratorIsStable`、`Cuda0AndCuda1HaveIndependentDefaults` |
| 6 | 显式 vs 默认 fallback | `ResolveGenerator`（`generator_impl.cc:200-205`），调用方 `init.cc:25`、`functional.cc:96/106/116`、`autograd/dropout.cc:20` | `NullGeneratorFallsBackToDefault` |
| 7 | Dropout 公式正向/反向 | `cpu/dropout.cc:51-79`、`cuda/dropout.cu:30-44` | 6 个 `GeneratorDropoutOpTest.*` |
| 8 | `Module.Train/Eval` 递归 | `module.cc:288-296` | `ModuleTrainingTest.*` (4 case) |
| 9 | Cross-magic state 拒绝 | `CUDAGeneratorImpl::SetState` 校验 magic | `RejectsCpuStateBlob` |

### 5.3 未对齐项

> 5 项已知差异，均由 spec / 赛题显式允许或非目标。

| # | 差异 | 原因（Fact） | Decision |
|---|---|---|---|
| 1 | 与 PyTorch bit 不一致 | CPU mt19937_64 ≠ PyTorch mt19937；CUDA Philox 调用约定不同 | 赛题 §三 末段允许 |
| 2 | CPU vs CUDA 同 seed 输出不一致 | 引擎不同必然不同 | 赛题 §二(5) 明确允许 |
| 3 | dtype 仅 FP32 | random kernel `CHECK_EQ(dtype, kFLOAT32)` fail-fast | 非本期范围；解除路径 §7 |
| 4 | `nn::Dropout` 缺 `inplace` | autograd `Function::Apply` 契约不支持原位写 | 同 PyTorch 默认（out-of-place）；多一份显存可接受 |
| 5 | `Module.Train` 不跳 `__pp_*` | `StateDict()` 跳是为序列化去重；`Train()` 跳会让 PP 改别名时漏切 | 见决策 §1.3 之 4 |

---

## 6. 测试与可复现性（赛题 §四 之三）

### 6.1 编译

```bash
# CUDA Debug（提交版本）
PATH=/usr/local/cuda/bin:$PATH cmake -B build \
  -DBUILD_TEST=ON -DUSE_CUDA=ON -DUSE_NCCL=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j 4    # -j 不限会触发 cc1plus OOM, 见 PR-prep §10
ctest --test-dir build --output-on-failure

# CPU-only（本地无 GPU 时）
cmake -B build_cpu -DBUILD_TEST=ON -DUSE_CUDA=OFF -DUSE_NCCL=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5    # 兼容 CMake ≥4 与 gflags submodule
cmake --build build_cpu -j
ctest --test-dir build_cpu --output-on-failure
```

要求：`-DBUILD_TEST=ON` 必须显式打开（默认 OFF，`CMakeLists.txt:7`）。

### 6.2 期望 ctest 数字

完整 ctest（train-server CUDA Debug，HEAD `1307669`）：

```
98% tests passed, 10 tests failed out of 532
(534 total - 2 Disabled = 532 ran; 510 pass / 10 fail / 12 skip)
Total Test time (real) = 90.61 sec
```

Generator 子集（`-R 'Generator|ModuleTrainingTest'`）：

```
100% tests passed, 0 tests failed out of 68
(34 CPU + 34 CUDA; 1 SKIP for GeneratorDefaultMultiGpu on 1-GPU host)
```

**注意 filter 形式**：测试名是 gtest fixture 风格 `CPU/GeneratorBasicTest.*`，**不是** `test_generator_*` binary 名；早期 `9de4b0e` 提交的脚本 filter 写错过，`edc5265` 修复。

### 6.3 复现脚本（加分项 spec §9.1）

```bash
./scripts/check_reproducibility.sh build       # CUDA build
./scripts/check_reproducibility.sh build_cpu   # CPU-only build
```

实现：连续两次 `ctest -R 'Generator|ModuleTrainingTest'`，逐测试提取 `Test +#[0-9]+: <name> ... <Passed|Failed|Skipped>` verdict 行（grep + sed 砍时间字段），diff 两次结果，全 match 则 exit 0；空输出 fail-fast `ERROR: no Generator test verdicts captured` exit 3。

期望：

| 环境 | 输出 | 退出码 |
|---|---|---|
| Train-server CUDA build | `PASS: 68 test outcomes match across two ctest runs` | 0 |
| 本地 CPU-only build | `PASS: 34 test outcomes match across two ctest runs` | 0 |

### 6.4 完整 ctest 中 10 个失败的实证归因

> 这 10 个 fail **不是本 PR 引入的**。证据链如下，对比 fact / hypothesis / decision 显式标注。

**Fact**（观测）：

```
261, 333, 334, 336    CUDA/AutogradMatmulForwardTest.*
330, 331, 332         CUDA/AutogradMatmulBackwardTest.*
261                   CUDA/AutogradForwardTest.MatmulForward
383, 389, 390         CUDA/LoRATest.{LoRALinearMerge, GetLoRAModel, MergeAndUnload}
```

均 abort 于 `gemm.cu:41] Check failed: p.stride_a == 0LL (... vs. 0)`。

**Hypothesis**（推测）：非批量 matmul caller 仍传非零 stride；`5dfd4b2e`（chen, 2026-04-15）引入的 DCHECK 在 debug 拒绝。

**实证 1（静态归因）**：

- `git blame infini_train/src/kernels/cuda/common/gemm.cu` → DCHECK 来自 `5dfd4b2e`（2026-04-15）和 `fd7c3a1c`（2026-05-07），**两 commit 都在 `origin/master` 上、远早于本分支起点 2026-05-21**。
- `git diff master..HEAD --stat` + `git log master..HEAD -- 'gemm.cu' '*matmul*' '*lora*'` → 本 PR **不触碰**这些文件。

**实证 2（pristine master 上单独 build 复现）**：

train-server 上独立 `build-master/` 跑 `ctest`：

```
98% tests passed, 10 tests failed out of 403
Total Test time (real) = 78.86 sec
```

10 fail 测试名：

```
210                CUDA/AutogradForwardTest.MatmulForward
279, 280, 281      CUDA/AutogradMatmulBackwardTest.*
282, 283, 285      CUDA/AutogradMatmulForwardTest.*
332, 338, 339      CUDA/LoRATest.{LoRALinearMerge, GetLoRAModel, MergeAndUnload}
```

测试编号不同（master 无新增 Generator suite，总数少 129），但**测试名集合 1:1 完全相同**，失败签名 1:1 一致。

**Decision**：不属于本 PR scope；建议上游修非批量 matmul caller 或放宽 `gemm.cu:41` DCHECK。详细 console block 见 [PR-prep notes §10](superpowers/notes/2026-05-21-phase2-pr-prep.md#10-empirical-proof-pristine-upstream-master-has-the-same-10-failures-2026-05-21)。

12 skip 的 outcome 也已确认与本 PR 无关：1 个多 GPU 测试 + 1 个 TensorCopy 多设备 + 10 个 CPU `TransformerModuleTest`（基础设施限制）。

---

## 7. 风险与遗留问题

| 项 | 风险 | 状态 / 处理 |
|---|---|---|
| 上游 `gemm.cu:41` DCHECK | 任何 debug 全量 ctest 都会触发；CI 红 | 与本 PR 无关，已实证（§6.4）；建议上游修 |
| `nn::Dropout` 无 `inplace` | 显存多一份 | spec §10 已声明；后续 autograd 改造可解 |
| dtype 仅 FP32 | BF16/FP16 走 random kernel 会 fail-fast | spec §10 已声明；模板分发可解 |
| OMP 多线程 RNG | 单线程 + mutex 在多核 CPU 是性能瓶颈 | 赛题 §2.2 非目标；每线程 sub-stream + Philox jump-ahead 可解 |
| CUDA graph capture | offset 在 host mutex 推进，graph capture 路径需 external input | Phase 3 spec §2.6 已记 |
| LoRA `dropout` 字段未接线 | LoRA 模块未消费 `LoRAConfig::dropout` | Phase 3 spec §1.5 已记，挂 `nn::Dropout` 即可 |

### 7.1 后续可扩展方向

1. BF16/FP16 dtype 解锁（解 `CHECK_EQ` + 模板分发）
2. OMP 多线程 RNG（每线程 sub-stream + Philox jump-ahead）
3. CUDA graph capture friendly Philox snapshot
4. Bernoulli / randint / categorical（复用 `ResolveGenerator` + dispatcher）
5. ZeRO/分布式 RNG（per-rank `seed + rank` 派生）
6. LoRA `dropout` 字段挂 `nn::Dropout`

---

## 附录 A：相关 spec / notes

- 设计 spec：[`docs/superpowers/specs/2026-05-19-generator-abstraction-design.md`](superpowers/specs/2026-05-19-generator-abstraction-design.md)
- Phase 3 Dropout spec：[`docs/superpowers/specs/2026-05-21-generator-phase3-dropout-design.md`](superpowers/specs/2026-05-21-generator-phase3-dropout-design.md)
- 实测留档：[`docs/superpowers/notes/2026-05-21-phase2-pr-prep.md`](superpowers/notes/2026-05-21-phase2-pr-prep.md)（§8 PR body 模板、§9 Phase 3+4 数据、§10 master 实证 console block）
- 复现脚本：[`scripts/check_reproducibility.sh`](../scripts/check_reproducibility.sh)
