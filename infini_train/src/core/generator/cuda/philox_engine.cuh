#pragma once

// =============================================================================
// Adapted from PyTorch (aten/src/ATen/core/PhiloxRNGEngine.h).
// Source: https://github.com/pytorch/pytorch/blob/bf6b7fb48c55069847a0887eb84b0781e0c9a838/aten/src/ATen/core/PhiloxRNGEngine.h
//
// The verbatim copyright notice and BSD-3-Clause license text from PyTorch's
// top-level LICENSE file follows. This is reproduced here to satisfy clauses
// (1) and (3) of the BSD-3-Clause license.
//
// ----- BEGIN VERBATIM PyTorch LICENSE -----
//
// From PyTorch:
//
// Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
// Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
// Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
// Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
// Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
// Copyright (c) 2011-2013 NYU                      (Clement Farabet)
// Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
// Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
// Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
//
// From Caffe2:
//
// Copyright (c) 2016-present, Facebook Inc. All rights reserved.
//
// All contributions by Facebook:
// Copyright (c) 2016 Facebook Inc.
//
// All contributions by Google:
// Copyright (c) 2015 Google Inc.
// All rights reserved.
//
// All contributions by Yangqing Jia:
// Copyright (c) 2015 Yangqing Jia
// All rights reserved.
//
// All contributions by Kakao Brain:
// Copyright 2019-2020 Kakao Brain
//
// All contributions by Cruise LLC:
// Copyright (c) 2022 Cruise LLC.
// All rights reserved.
//
// All contributions by Tri Dao:
// Copyright (c) 2024 Tri Dao.
// All rights reserved.
//
// All contributions by Arm:
// Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates
//
// All contributions from Caffe:
// Copyright(c) 2013, 2014, 2015, the respective contributors
// All rights reserved.
//
// All other contributions:
// Copyright(c) 2015, 2016 the respective contributors
// All rights reserved.
//
// Caffe2 uses a copyright model similar to Caffe: each contributor holds
// copyright over their contributions to Caffe2. The project versioning records
// all such contribution and copyright details. If a contributor wants to further
// mark their specific copyright on a particular contribution, they should
// indicate their copyright solely in the commit message of the change when it is
// committed.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
//    and IDIAP Research Institute nor the names of its contributors may be
//    used to endorse or promote products derived from this software without
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ----- END VERBATIM PyTorch LICENSE -----
//
// Implementation note:
//   PyTorch's upstream class is host+device (C10_HOST_DEVICE) and uses
//   std::array<uint32_t, 4>/<uint32_t, 2> for counter/key/output buffers,
//   plus __umulhi() inside __CUDA_ARCH__ for the 32x32->64 multiply.
//   This vendored copy keeps the same Philox-4x32-10 algorithm and constants
//   (kPhiloxSA/SB, kPhilox10A/B, 10 rounds) but exposes a __device__-only
//   surface tailored for InfiniTrain's CUDA generator path:
//     * constructor takes (seed, subsequence, offset)
//     * operator() returns a uint32_t
//     * Uniform01() returns float in [0, 1) via 24-bit mantissa stuffing
//   The state representation (uint32_t[4] counter, uint32_t[2] key,
//   uint32_t[4] output, uint32_t state index) mirrors PyTorch's so that, given
//   the same (seed, subsequence, offset), the i-th uint32_t produced by
//   operator() is bit-identical to PyTorch's philox_engine::operator()(10).
// =============================================================================

#include <cstdint>

#include <cuda_runtime.h>

namespace infini_train::core::cuda {

class Philox4_32 {
public:
    __device__ inline Philox4_32(uint64_t seed, uint64_t subsequence, uint64_t offset) {
        key_[0] = static_cast<uint32_t>(seed);
        key_[1] = static_cast<uint32_t>(seed >> 32);
        counter_[0] = 0;
        counter_[1] = 0;
        counter_[2] = static_cast<uint32_t>(subsequence);
        counter_[3] = static_cast<uint32_t>(subsequence >> 32);
        output_[0] = 0;
        output_[1] = 0;
        output_[2] = 0;
        output_[3] = 0;
        state_ = 0;
        IncrN(offset);
    }

    // Produces a unique 32-bit pseudo-random number on every invocation.
    // Bookkeeps state to avoid waste, matching PyTorch's philox_engine::operator()(10).
    __device__ inline uint32_t operator()() {
        if (state_ == 0) {
            uint32_t counter[4] = {counter_[0], counter_[1], counter_[2], counter_[3]};
            uint32_t key[2] = {key_[0], key_[1]};
            Rand10(counter, key, output_);
            Incr();
        }
        const uint32_t ret = output_[state_];
        state_ = (state_ + 1) & 3u;
        return ret;
    }

    // Returns float in [0, 1) using the top 24 bits as the mantissa.
    __device__ inline float Uniform01() {
        return static_cast<float>((*this)() >> 8) * (1.0f / 16777216.0f);
    }

private:
    // Skips n 128-bit numbers in the subsequence. Mirrors PyTorch's incr_n.
    __device__ inline void IncrN(uint64_t n) {
        const uint32_t nlo = static_cast<uint32_t>(n);
        uint32_t nhi = static_cast<uint32_t>(n >> 32);
        counter_[0] += nlo;
        if (counter_[0] < nlo) {
            ++nhi;
            counter_[1] += nhi;
            if (nhi != 0) {
                if (nhi <= counter_[1]) {
                    return;
                }
            }
        } else {
            counter_[1] += nhi;
            if (nhi <= counter_[1]) {
                return;
            }
        }
        if (++counter_[2]) {
            return;
        }
        ++counter_[3];
    }

    // Skips one 128-bit number in the subsequence. Mirrors PyTorch's incr.
    __device__ inline void Incr() {
        if (++counter_[0]) {
            return;
        }
        if (++counter_[1]) {
            return;
        }
        if (++counter_[2]) {
            return;
        }
        ++counter_[3];
    }

    // 32x32->64 multiply: returns the low 32 bits, writes the high 32 bits to *result_high.
    static __device__ inline uint32_t MulHiLo32(uint32_t a, uint32_t b, uint32_t *result_high) {
        *result_high = __umulhi(a, b);
        return a * b;
    }

    static __device__ inline void SingleRound(const uint32_t ctr[4], const uint32_t in_key[2], uint32_t out[4]) {
        uint32_t hi0 = 0;
        uint32_t hi1 = 0;
        const uint32_t lo0 = MulHiLo32(kPhiloxSA, ctr[0], &hi0);
        const uint32_t lo1 = MulHiLo32(kPhiloxSB, ctr[2], &hi1);
        out[0] = hi1 ^ ctr[1] ^ in_key[0];
        out[1] = lo1;
        out[2] = hi0 ^ ctr[3] ^ in_key[1];
        out[3] = lo0;
    }

    // 10-round Philox-4x32 (matches PyTorch's rand(counter, key, /*n_rounds=*/10)).
    static __device__ inline void Rand10(uint32_t counter[4], uint32_t key[2], uint32_t out[4]) {
        uint32_t tmp[4];
        for (int round = 0; round < 9; ++round) {
            SingleRound(counter, key, tmp);
            counter[0] = tmp[0];
            counter[1] = tmp[1];
            counter[2] = tmp[2];
            counter[3] = tmp[3];
            key[0] += kPhilox10A;
            key[1] += kPhilox10B;
        }
        SingleRound(counter, key, out);
    }

    // Philox-4x32 constants (Salmon et al. 2011, Random123).
    // Mirrors PyTorch's kPhilox10A / kPhilox10B / kPhiloxSA / kPhiloxSB.
    static constexpr uint32_t kPhilox10A = 0x9E3779B9u;
    static constexpr uint32_t kPhilox10B = 0xBB67AE85u;
    static constexpr uint32_t kPhiloxSA = 0xD2511F53u;
    static constexpr uint32_t kPhiloxSB = 0xCD9E8D57u;

    uint32_t counter_[4];
    uint32_t output_[4];
    uint32_t key_[2];
    uint32_t state_;
};

} // namespace infini_train::core::cuda
