# InfiniTrain Generator 抽象 — PyTorch 对齐报告

> 2026 春季 AI 大赛 Generator 选题。分支 `feat/generator-phase1`。
> Spec：`docs/superpowers/specs/2026-05-19-generator-abstraction-design.md`、`2026-05-21-generator-phase3-dropout-design.md`。

## 1. 概述与设计参考

为 InfiniTrain 引入与 PyTorch `c10::Generator` 等价抽象：句柄+多态 Impl+CPU/CUDA 后端+默认池+全局 seed+算子接入+Dropout 链路。参考赛题§六四链接：[`torch.Generator`](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#generator)、[`Generator.h`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/Generator.h)、[`CPUGeneratorImpl.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/CPUGeneratorImpl.cpp)、[`CUDAGeneratorImpl.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp) 与 [Randomness Notes](https://docs.pytorch.org/docs/stable/notes/randomness.html#pytorch-random-number-generator)。映射：`Generator`↔`c10::Generator`；`GeneratorImpl`↔`c10::GeneratorImpl`；`CPUGeneratorImpl`(mt19937_64)↔`at::CPUGeneratorImpl`；`CUDAGeneratorImpl`(Philox4_32+offset 16B)↔`at::CUDAGeneratorImpl`；`GeneratorImplRegistry`↔`at::detail::DefaultGenerator`；`manual_seed`↔`torch.manual_seed`。

## 2. 接口对照表

**句柄**(`generator.h`)：`Generator(Device)`(L18)↔`Generator(device=...)`；`ManualSeed/Seed/InitialSeed`(L25-27)↔`manual_seed/seed/initial_seed`(命名差异)；`GetState/SetState`(L29-30 返 `vector<uint8_t>`)↔`get_state/set_state`(byte buffer vs Tensor)；`device()`(L32)↔`gen.device`；浅拷贝(L20-23 `shared_ptr` ↔ `intrusive_ptr`)。

**基类**(`generator_impl.h`)：`mutex/device/StateMagic`+抽象 `SetCurrentSeed/CurrentSeed/InitialSeed/GetState/SetState/Reseed`(L25-38) 对齐 `c10::GeneratorImpl`；base `Seed()`(`generator_impl.cc:27`) 调 `Reseed` 不改 `initial_seed_`。

**默认池+seed**：`default_generator(dev)`(`generator.h:46`、`generator_impl.cc:181`)↔`globalContext().defaultGenerator`，多次取共享(once_flag L167-194)；`manual_seed`→`ResetAllSeeds`(L96) 同步 CPU+已/未初始化 CUDA↔`torch.manual_seed`；`manual_seed_cuda`→`ResetCudaSeeds`(L116)↔`torch.cuda.manual_seed_all`；未初始化经 `last_user_seed_`+`LastUserSeedOrRandom`(L61-69) 首访取记忆 seed。

**Module training**(`module.h`)：`Train(bool)/Eval()/IsTraining()`(L92-94)↔`Module.train/eval/training`；`module.cc:288-296` 递归 `modules_` 仅跳 `nullptr`。**`__pp_*` 不跳过**：`StateDict()` 跳 `__pp_*` 是序列化去重需求(`TransformerModel` 把 `__pp_chunk_*` 内 layer 别名暴露主名)；`Train()` 幂等 setter 无去重需求；跳过则未来 PP 改别名策略静默漏切 chunk 内 Dropout 的 eval。

**Dropout**：`autograd::Dropout`(`autograd/dropout.cc:22-27`，`saved_tensors_={mask}`)↔`aten::native_dropout` 多返回；`nn::Dropout`(`nn/modules/dropout.h:16`，`IsTraining()` 门控)↔`nn.Dropout`，**缺 `inplace`**；`nn::function::Dropout`(`functional.cc:111`)↔`nn.functional.dropout`；mask `kUINT8`(`cpu/dropout.cc:30`)↔`bool`，语义等价。

**Random**：`nn::function::Rand/Randn`(`functional.cc:91/101`)↔`torch.rand/randn`；`nn::init::Uniform/Normal/KaimingUniform`(`init.cc:91/21/78`)↔`nn.init.uniform_/normal_/kaiming_uniform_`，全部走 dispatcher；**dtype 仅 FP32**(kernel `CHECK_EQ(kFLOAT32)` fail-fast)。

## 3. 行为对齐情况

3.1 **同 seed 复现**：`test_seed.cc::ManualSeedReseedsState`(L14)、`test_ops_uniform.cc::SameSeedSameOutput`(L27)、`test_ops_normal.cc::SameSeedSameOutput`(L28)、`test_ops_dropout.cc::SameSeedSameMask`(L49)；`INFINI_TRAIN_REGISTER_TEST` 自动 CPU+CUDA 双实例化。

3.2 **状态推进 trap**：连续两次同长序列必须不同。`test_seed.cc::DifferentSeedsDifferState`(L23)、`test_ops_uniform.cc::ConsecutiveCallsAdvanceState`(L40)。单线程+锁(`uniform_random.cc:22`) 修复旧 OMP 路径"每线程重新构造 mt19937 状态不推进"bug。

3.3 **State 往返+跨格式拒绝**：`test_state.cc::GetSetStateRoundtrip`(L14)。CPU 校验 `cpu_only/test_state_validation.cc` 4 case `TruncatedHeader/BadMagic/BadVersion/PayloadSizeMismatch`(L27-48)；CUDA `cuda_only/test_cuda_state_validation.cc` 6 case 含 `RoundtripPreservesSeedOffsetInitialSeed`、`RejectsCpuStateBlob`(L73 cross-magic 拒喂 CPU state blob)。

3.4 **InitialSeed 不被 Seed() 覆盖**：`test_initial_seed.cc::SeedDoesNotChangeInitialSeed`(L12)。`generator_impl.cc:27-32`+`cpu_generator_impl.cc:44-47`。

3.5 **默认池一致性**：`test_generator_basic.cc::ConstructionAttachesDevice`(L14)、`test_default.cc::DefaultGeneratorIsStable`(L15)。多 GPU：`cuda_only/test_default_multi_gpu.cc::Cuda0AndCuda1HaveIndependentDefaults`(L27)，1-GPU 主机自动 SKIP。

3.6 **显式 vs 默认**：`test_dispatch.cc::NullGeneratorFallsBackToDefault`(L18)。`ResolveGenerator`(`generator_impl.cc:200-205`) 被 `init.cc:25`、`functional.cc:96/106/116`、`autograd/dropout.cc:20` 调用。

3.7 **Dropout 行为**：`out_i=mask_i?x_i/(1-p):0`、`dx_i=mask_i?dy_i/(1-p):0`，`mask_i∈{0,1}` `kUINT8`；`cpu/dropout.cc:51-58/77-79`、`cuda/dropout.cu:30-36/40-44`。`test_ops_dropout.cc` 6 case：`SameSeedSameMask`(49)、`MaskKeepRateSane`(65 保留率`[1-p±0.05]`)、`OutputScaleCorrect`(86)、`BackwardEqualsMaskScale`(104)、`EvalGateBypassesDropout`(121)、`FunctionalTrainingFalseBypasses`(129)。

3.8 **Module.Train/Eval 递归**：`test_module_training.cc` 4 case `DefaultIsTrainingTrue`(32)/`EvalRecursesIntoChildren`(37)/`RecursesIntoPpPrefixedChildren`(55)/`NullChildSkipped`(68)。`module.cc:288-296`。

## 4. 未对齐项及原因

1. **同 seed 与 PyTorch bit 不一致**：CPU mt19937_64、CUDA Philox4_32 与 PyTorch 引擎参数不同；spec §2.2 与赛题§五均不要求 bit 一致。
2. **CPU vs CUDA 同 seed 不一致**：引擎不同必然不同；spec §6 显式声明，与 PyTorch 行为一致。
3. **相对旧 `init.cc` 默认序列变化**：旧 `static std::mt19937 gen(kRandomSeed=42)`；本实现改 `mt19937_64`，未 `manual_seed` 时 `random_device` 兜底(`generator_impl.cc:18-23/61-69`)；spec §2.3 已声明。
4. **dtype 仅 FP32**：随机 kernel `CHECK_EQ(kFLOAT32)` fail-fast(`uniform_random.cc:18`、`cuda/uniform_random.cu:40`、`cpu/dropout.cc:22`、`cuda/dropout.cu:53`)。
5. **`nn::Dropout` 缺 `inplace`**：`nn/modules/dropout.h:20` 仅 `(p, generator)`，等价但显存多一份。

## 5. 可扩展方向

1. LoRA `LoRAConfig::dropout` 接线（spec §5.5、Phase 3 §1）。
2. BF16/FP16：解除 `CHECK_EQ` + 模板分发（参考 `Cast`/`Fill`）。
3. OMP 多线程 RNG（spec §2.2 非目标）：每线程独立 sub-stream + Philox jump-ahead。
4. CUDA graph capture 友好 Philox state snapshot（Phase 3 spec §2.6，offset 作 graph external input）。
5. Bernoulli/randint：复用 `ResolveGenerator`+dispatcher。
6. ZeRO/分布式 RNG（spec §10）：rank 派生 `seed+rank`。

## 6. 验证清单与复现说明

CUDA：`-DUSE_CUDA=ON -DUSE_NCCL=ON`；CPU-only 加 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` 兼容 gflags。运行 `ctest --test-dir <build> --output-on-failure -R '^test_generator_'`。

期望(见 `docs/superpowers/notes/2026-05-21-phase2-pr-prep.md`)：train-server `USE_CUDA=ON` **519 pass / 10 fail / 1 skip 共 530**；Generator suite 40+ pass/0 fail/1 skip(Multi-GPU SKIP)。10 fail 是 pre-existing upstream `gemm.cu:41` DCHECK(commit `5dfd4b2e` 2026-04-15、`fd7c3a1c` 2026-05-07)。CPU-only 本地 **298 pass / 1 fail**(`third_party/glog::log_severity_conversion`)。加分项 `scripts/check_reproducibility.sh`(spec §9.1)：`./scripts/check_reproducibility.sh build`，0 表示可复现。
