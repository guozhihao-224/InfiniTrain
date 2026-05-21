# Phase 2 PR-prep notes (2026-05-21)

User decision: 等 Phase 3+4 完成后再统一开 PR。这份文件保存 Phase 2 收尾的实测数据，避免开 PR 时重新跑。

Branch: `feat/generator-phase1` (Phase 2 commits 已 push 到 `guozhihao/feat/generator-phase1`)

---

## 1. Commit sequence (`git log --oneline master..HEAD`, 24 commits)

Phase 2 (10):

```
a76bdeb test(generator): cover multi-GPU default generator independence       # Task 8
ef9ee12 test(generator): activate CUDA branch for kaiming/dispatch ops        # Task 7
5747ce5 feat(generator): add CUDA NormalRandom kernel (Box-Muller, stream-bound)  # Task 6
bc8ed2d feat(generator): add CUDA UniformRandom kernel (Philox, stream-bound)     # Task 5
d09cb16 feat(generator): CUDAGeneratorImpl state serialization + cross-magic reject  # Task 4
048263e style(generator): drop unused <stdexcept> + tighten EXPECT_DEATH regex     # Task 3 review fix
91c3a2a fix(kernels/cuda): add missing <numeric>/<functional> for std::accumulate  # Task 3a
2bad603 feat(generator): add CUDAGeneratorImpl (Philox seed+offset) + factory      # Task 3
93994c1 feat(generator): vendor PyTorch Philox4_32 device engine (BSD-3)           # Task 2
0477bbe test(generator): wire tests/generator/cuda_only/ into CMake                # Task 1
```

Phase 1 (14):

```
dc26345 refactor(example): drop dead kRandomSeed and unused std::mt19937 in checkpoint loaders
3b678cf refactor(nn/init): route Uniform/Normal/KaimingUniform through Generator + dispatcher
af79927 feat(generator): add CPU NormalRandom kernel
8678fae feat(generator): add CPU UniformRandom kernel + dispatcher integration
07ab6a3 feat(generator): add ResolveGenerator helper for op layer
cd3e52f feat(generator): add CPU state serialization with magic/version header
3dfeb98 feat(generator): implement default pool, manual_seed, default_generator()
f5ad92e feat(generator): add Generator PImpl handle bound to Registry::Create()
8b2472e feat(generator): add CPUGeneratorImpl with mt19937_64 engine + tests
ad16453 feat(generator): add GeneratorImpl base + Registry skeleton
3e9d7f1 test(generator): add tests/generator scaffold with placeholder gtest
604b33f chore(third_party): add googletest submodule
d2db9b1 feat(generator): implement Phase 1 of generator abstraction
5809938 chore: ignore .worktrees/ for local subagent isolation
```

---

## 2. Train-server full ctest result (USE_CUDA=ON, USE_NCCL=ON)

```
98% tests passed, 10 tests failed out of 509
cpu     =  18.62 sec*proc (215 tests)
cuda    =  58.51 sec*proc (220 tests)
Total Test time (real) =  86.26 sec
```

### Generator suite (Phase 2 surface): **40 pass / 1 skip / 0 fail**

The 1 skip is the multi-GPU test on a 1-GPU host:

```
512/512 Test #512: GeneratorDefaultMultiGpu.Cuda0AndCuda1HaveIndependentDefaults ... ***Skipped   0.16 sec
```

Tail of the generator block (proof of CUDA path coverage):

```
502/512 CUDAGeneratorImplTest.ManualSeedSetsCurrentInitialAndZeroOffset    Passed   0.12 sec
503/512 CUDAGeneratorImplTest.NextPhiloxStateAdvancesOffset                Passed   0.08 sec
504/512 CUDAGeneratorImplTest.ReseedThroughBaseClassSeed                   Passed   0.09 sec
505/512 CUDAGeneratorImplTest.IndexOutOfRangeAborts                        Passed   0.37 sec
506/512 CudaStateValidation.RoundtripPreservesSeedOffsetInitialSeed        Passed   0.08 sec
507/512 CudaStateValidation.TruncatedHeader                                Passed   0.09 sec
508/512 CudaStateValidation.BadMagic                                       Passed   0.08 sec
509/512 CudaStateValidation.BadVersion                                     Passed   0.08 sec
510/512 CudaStateValidation.PayloadSizeMismatch                            Passed   0.08 sec
511/512 CudaStateValidation.RejectsCpuStateBlob                            Passed   0.08 sec
512/512 GeneratorDefaultMultiGpu.Cuda0AndCuda1HaveIndependentDefaults      Skipped  0.16 sec
```

### 10 failing tests (ALL pre-existing upstream issues, NOT Phase 2 regressions)

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

All abort on the same DCHECK in `infini_train/src/kernels/cuda/common/gemm.cu:41`:

```
F20260521 08:18:55.833890 ... gemm.cu:41] Check failed: p.stride_a == 0LL (12 vs. 0)
F20260521 08:18:56.423182 ... gemm.cu:41] Check failed: p.stride_a == 0LL (128 vs. 0)
```

`git blame` attribution:

```
5dfd4b2e (chen 2026-04-15) DCHECK_EQ(p.stride_a, 0LL);  -- on origin/master
fd7c3a1c (chen 2026-05-07) gemm refactor               -- on origin/master
```

`git diff origin/master..HEAD --stat` shows Phase 2 does NOT touch `gemm.cu`/`matmul.cu`/anywhere on this code path. The DCHECK was introduced upstream on 2026-04-15 (≥ 1 month before Phase 2 work began on 2026-05-21). The non-batched matmul callers pass non-zero strides, which trips the DCHECK in debug builds.

→ Out of scope for this PR. Surface to upstream as a separate issue if needed.

---

## 3. Local CPU-only build (Step 4 self-check)

```bash
cmake -S . -B build_cpu -DBUILD_TEST=ON -DUSE_CUDA=OFF -DUSE_NCCL=OFF \
     -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build_cpu -j 8
ctest --test-dir build_cpu --output-on-failure
```

### CMake quirk (worth noting in PR body if anyone tries to reproduce locally)

Without `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`, configure fails on `third_party/gflags/CMakeLists.txt:73 (cmake_minimum_required)` because modern CMake (≥4.x) removed `<3.5` compatibility. This is not Phase 2's concern; a Phase 2-independent fix would be bumping gflags submodule or adding the policy override at the root CMakeLists.

### Result: **237 pass / 11 skip / 1 fail (third-party glog)**

Build tree correctly excludes `test_generator_cuda_only` (gated by root `if(USE_CUDA)`), confirmed by:

```
$ ls build_cpu/tests/generator/
test_generator_cpu[1]_include.cmake
test_generator_cpu_only[1]_include.cmake
test_generator_cuda[1]_include.cmake     # <- parameterized fixture, CPU portion only
# (no test_generator_cuda_only)
```

Single failure: `20 - log_severity_conversion` — defined in `third_party/glog/CMakeLists.txt:817`. Not InfiniTrain code; not Phase 2.

---

## 4. Phase 2 SKIP-marker self-check (Task 9 Step 2)

```bash
grep -rn 'is Phase 2\|CUDA random kernels are Phase 2\|CUDA NormalRandom is Phase 2\|CUDA UniformRandom is Phase 2' tests/
```

Empty. All Phase 1 SKIP guards from `tests/generator/test_ops_*.cc`, `test_ops_kaiming.cc`, `test_dispatch.cc` removed in Phase 2 (Tasks 5/6/7).

## 5. mt19937 surface self-check (Task 9 Step 3)

```bash
grep -rn "mt19937" example/ infini_train/ | grep -v "third_party\|build/"
```

```
infini_train/include/core/generator/cpu_generator_impl.h:24:    std::mt19937_64 &engine() { return engine_; }
infini_train/include/core/generator/cpu_generator_impl.h:35:    std::mt19937_64 engine_{0};
```

Only `mt19937_64` (the engine type) appears, only inside `CPUGeneratorImpl` — exactly as spec mandates. No bare `std::mt19937` anywhere; no `kRandomSeed` constant survives.

---

## 6. Spec §2.1 必达目标 — Phase 2 status

| 目标 | Phase 2 落地点 | 状态 |
|------|----------------|------|
| Generator 抽象统一 | (Phase 1) | ✓ |
| CPU + CUDA 后端 | Task 3 (CUDAGeneratorImpl + factory), Task 4 (state) | ✓ |
| 默认池（CPU + 每 CUDA 设备） | Task 3 + Task 8 验证 | ✓ |
| 全局 manual_seed 走通 CUDA | Task 3 工厂注册让 §4.6 进入 CUDA 分支 | ✓ |
| 算子改造（CUDA） | Task 5 UniformRandom, Task 6 NormalRandom, Task 7 unblock kaiming/dispatch | ✓ |
| 测试矩阵 (spec §7.2) | Task 5/6/7 解 SKIP；Task 4 cuda_only state；Task 8 多 GPU | ✓ |
| 报告 | — | Phase 4 |

## 7. 已知 Phase 2 末态边界（不算回归，spec §10 已声明）

1. Dropout 链路缺位 → Phase 3
2. `nn::function::Rand/Randn` 缺位 → Phase 3
3. Dtype 仅 FP32（BF16/FP16/FP64 在 random kernel `CHECK_EQ` fail-fast）
4. OMP 多线程 RNG 未优化（spec §2.2 非目标）
5. PyTorch 数值 bit 不一致（仅语义对齐）→ Phase 4 报告

---

## 8. 开 PR 时的可粘贴模板（等 Phase 3+4 完成后再用）

```markdown
## Summary
- Phase 1: introduce Generator/GeneratorImpl PImpl, CPU backend (mt19937_64 + state header), default pool, manual_seed, ResolveGenerator, CPU UniformRandom/NormalRandom kernels, route nn::init::{Uniform,Normal,KaimingUniform} through dispatcher.
- Phase 2: vendor PyTorch Philox4_32 (BSD-3), CUDAGeneratorImpl (seed+offset), state serialization with magic/version + cross-magic reject, CUDA UniformRandom (Philox) and NormalRandom (Box-Muller), stream-bound launches, multi-GPU default generator independence.
- Phase 3: `nn::Module::Train/Eval/IsTraining` (recurses through `__pp_*`), CPU + CUDA `DropoutForward/DropoutBackward` (mask `kUINT8`, scale `1/(1-p)`, CUDA Philox stream-bound + p==0 fast path), `autograd::Dropout` Function, `nn::Dropout` module (IsTraining gated), `nn::function::{Rand,Randn,Dropout}` free helpers.
- Phase 4: full technical report at `docs/generator-design.md` mapping 1:1 to contest §四 — §1 functional correctness verification, §2 PyTorch alignment & behavior analysis, §3 test & reproducibility instructions, §4 §五 acceptance checklist; plus `scripts/check_reproducibility.sh` driving two ctest sweeps and diffing per-test verdicts; plus master-empirical-proof addendum confirming the 10 failures are upstream-only.

## Test plan
- train-server full ctest at HEAD `1307669` (USE_CUDA=ON USE_NCCL=ON Debug): **510 pass / 10 fail / 12 skip / 532 ran (534 - 2 Disabled)** — the 10 failures are the pre-existing upstream `gemm.cu:41` DCHECK aborts (unrelated; commit attribution in §2 below).
- Generator suite alone (`-R 'Generator|ModuleTrainingTest'`): **68 pass / 0 fail / 1 skip** (multi-GPU on 1-GPU host). 34 CPU + 34 CUDA, including the new `GeneratorStateTest.GetSetStateRestoresSequence` which closes spec §三(3).
- `scripts/check_reproducibility.sh build` on train-server CUDA build: `PASS: 68 test outcomes match across two ctest runs`, exit 0.
- Local CPU-only build (USE_CUDA=OFF): `cuda_only/` excluded; `scripts/check_reproducibility.sh build_cpu` reports `PASS: 34 test outcomes match across two ctest runs`, exit 0.
- Phase 4 alignment report cites file:line + test names for every claim; reproducibility script handles empty ctest output explicitly.

## Out of scope (filed/notable)
- gemm.cu:41 `DCHECK_EQ(p.stride_a, 0LL)` introduced by 5dfd4b2e (2026-04-15, upstream): non-batched matmul callers pass non-zero strides, tripping the DCHECK in debug. Pre-existing, surfaces in 7 Matmul + 3 LoRA tests.
- third_party/gflags requires `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` on modern CMake.
- BF16/FP16 dropout, LoRA `dropout` field wiring, PyTorch bit-for-bit reproducibility, CUDA graph capture-friendly Philox snapshot — all listed as future work in `docs/generator-design.md` §2.5.
```

---

## 9. Phase 3 + Phase 4 addendum (2026-05-21)

### Commit additions since section 1 (33 commits total since `master`)

Phase 4 (2):

```
edc5265 fix(scripts): correct reproducibility filter regex + handle empty ctest output
9de4b0e docs(generator): add Phase 4 PyTorch alignment report + reproducibility script
```

Phase 3 (7):

```
4709bca feat(nn/function): add Dropout free fn + integration test (mask/scale/backward/eval)
9db2167 feat(nn/function): add Rand/Randn helpers (FP32, dispatcher-routed)
a9a64b3 feat(nn/modules): add Dropout module gated by Module::IsTraining()
5bf1e43 feat(autograd): add Dropout Function (multi-return, saved mask)
a627e9a feat(kernels/cuda): add DropoutForward/DropoutBackward (Philox, stream-bound)
6eb259e feat(kernels/cpu): add DropoutForward/DropoutBackward (mask kUINT8, scale 1/(1-p))
d2a986d feat(nn/module): add Train/Eval/IsTraining with __pp_*-inclusive recursion
```

### Train-server ctest at HEAD `1307669` (Phases 1+2+3+4, USE_CUDA=ON USE_NCCL=ON Debug)

Re-run on 2026-05-21 after the state-realignment test was added:

```
98% tests passed, 10 tests failed out of 532
(534 total - 2 Disabled = 532 ran; 10 fail / 12 Skipped / 510 pass)
Total Test time (real) = 90.61 sec
```

Generator suite (filter `Generator|ModuleTrainingTest`):

```
100% tests passed, 0 tests failed out of 68
(34 CPU + 34 CUDA; 1 multi-GPU SKIP on this 1-GPU host)
```

Reproducibility script on train-server CUDA build:

```
$ ./scripts/check_reproducibility.sh build
PASS: 68 test outcomes match across two ctest runs
```

The 10 failures are the same `gemm.cu:41` upstream DCHECK aborts catalogued in §2 — confirmed unchanged before/after Phase 3+4 (commit attribution `5dfd4b2e`/`fd7c3a1c` predates this branch). The 12 Skipped are the multi-GPU generator test (1) + a TensorCopy multi-device test (1) + 10 CPU `TransformerModuleTest.*` (always skipped on this build).

Phase 3+4 newcomers (all green on both backends):
- `GeneratorStateTest.{GetSetStateRoundtrip,GetSetStateRestoresSequence}` ← spec §三(3) full save→draw→restore→draw cycle
- `ModuleTrainingTest.{DefaultIsTrainingTrue,EvalRecursesIntoChildren,RecursesIntoPpPrefixedChildren,NullChildSkipped}`
- `GeneratorDropoutOpTest.{SameSeedSameMask,MaskKeepRateSane,OutputScaleCorrect,BackwardEqualsMaskScale,EvalGateBypassesDropout,FunctionalTrainingFalseBypasses}`

### Phase 4 verification

- `docs/generator-design.md`: full technical report restructured 1:1 to contest §四 — §0 overview, §1 functional correctness verification (per §三(1)-(5) sub-clauses, with file:line + test-name citations for every claim), §2 PyTorch alignment & behavior analysis (interface tables + 9 aligned items + 6 unaligned items + 6 future directions), §3 test/reproducibility (build commands + expected ctest numbers + script usage + master empirical attribution), §4 §五 acceptance checklist with file references.
- `scripts/check_reproducibility.sh build_cpu` (local CPU-only): `PASS: 34 test outcomes match across two ctest runs`, exit 0.
- Script error-path verified: stale build (no Generator tests) → `ERROR: no Generator test verdicts captured`, exit 3 (no longer aborts under pipefail).
- Initial filter regex shipped in `9de4b0e` was wrong (`^test_generator_` matched executable names, not gtest names); fixed in follow-up `edc5265` to match `Generator|ModuleTrainingTest` + tightened verdict regex + `|| true` against empty grep. Reviewer lesson: structural `bash -n` PASS is necessary but not sufficient — script-quality review must run the script end-to-end.

### Train-server build hygiene (encountered during Phase 4 re-verification)

- `third_party/eigen` working tree was missing on train-server; rsync-uploaded directly (20 MB / 1800 files) since the box's network couldn't fetch the submodule. The remaining submodules (`gflags`, `glog`, `googletest`) had intact working trees; only their `.git` gitlinks were broken, which doesn't affect the build (CMake `add_subdirectory` only needs sources).
- CMake configure required `PATH=/usr/local/cuda/bin:$PATH` and `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` because nvcc wasn't on the non-interactive shell PATH.
- Tests are gated by `option(BUILD_TEST OFF)` (CMakeLists.txt:7); Phase 4 re-build needed `-DBUILD_TEST=ON` after the wipe.
- One transient parallel-build hiccup on `test_autograd_cpu` (Error 2 with no compile-error output) cleared on the next `cmake --build` invocation — standard `-j` race, not a code issue.

---

## 10. Empirical proof: pristine upstream master has the same 10 failures (2026-05-21)

To rule out regression entirely, built **upstream `InfiniTensor:master`** at SHA `a917760` ("feat(docs): add generator abstraction design and specifications") in a separate `build-master/` tree on train-server and ran the full ctest. SHA `a917760` is the parent of every Phase 1+2+3+4 commit on this branch (`git merge-base master HEAD == a917760`), so it's the apples-to-apples baseline.

Method: `git -c submodule.recurse=false read-tree --reset -u a917760` after temporarily moving aside `third_party/*/.git` gitlink files (so the broken `.git/modules/` doesn't trip git's submodule sanity checks); rsync re-populated `third_party/eigen/` afterwards (`read-tree -u` had cleared its untracked content). Build with `cmake --build build-master -j 4` (lower parallelism: `-j` triggered `c++: fatal error: Killed signal terminated program cc1plus` — host OOM, not a code issue).

**Result on master `a917760` (`USE_CUDA=ON USE_NCCL=ON Debug`, `cmake -B build-master -DBUILD_TEST=ON`):**

```
98% tests passed, 10 tests failed out of 403
Total Test time (real) = 78.86 sec
```

10 failures (verbatim):

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

All 10 abort with `gemm.cu:41] Check failed: p.stride_a == 0LL (... vs. 0)` — same DCHECK as catalogued in §2.

**Comparison to feat/generator-phase1:**

| Metric | master `a917760` | feat/generator-phase1 `1307669` |
|--------|------------------|-------------------------------|
| Total ctest size | 403 | 532 (= 534 − 2 Disabled) |
| Pass | 393 | 510 |
| Fail | 10 | 10 |
| Skip | (omitted; same 0 cuda + 12 cpu group as branch) | 12 |

The 10-test failure set matches **exactly** by gtest fixture/name (7 Matmul + 3 LoRA). Test numbers differ only because feat/generator-phase1 adds 129 more tests (the new generator suite). **The 10 failures pre-exist on upstream master and are not introduced by Phase 1+2+3+4.**

After empirical run, train-server was restored: branch `feat/generator-phase1` reset to `1307669`, working tree restored via `git -c submodule.recurse=false read-tree --reset -u feat/generator-phase1`, submodule `.git` gitlink files moved back from `.git.bak`, eigen content re-rsynced.
