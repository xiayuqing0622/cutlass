/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holdvr nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "kernel_backward.h"

#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
       #define CUDA_SAFE_CALL(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t result = (x);                                                                  \
        if (result != cudaSuccess)                                                                 \
        {                                                                                          \
            const char* msg = cudaGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
static constexpr int kMaxK = 128;

extern "C" int flash_attn_bwd(
                    void* grad_q,
                    void* grad_k,
                    void* grad_v,
                    void* grad_b,
                    void* output,
                    void* query,
                    void* key,
                    void* value,
                    void* bias,
                    void* lse,
                    void* grad_output,
                    void* delta,
                    int batch_size,
                    int seq_len,
                    int seq_len_kv,
                    int num_heads,
                    int head_dim,
                    int head_dim_value,
                    float softmax_scale,
                    bool causal)
{
    using Arch = cutlass::arch::Sm70;
    using Element = cutlass::half_t;

    static constexpr bool kSupports64x128 =
         Arch::kMinComputeCapability >= 80 ||
         (Arch::kMinComputeCapability >= 70 &&
         cutlass::sizeof_bits<Element>::value <= 16);
    static constexpr int kBlockSizeI = kSupports64x128 && kMaxK > 64 ? 128 : 64;
    static constexpr bool kIsHalf = cutlass::sizeof_bits<Element>::value <= 16;
    static constexpr bool kOutputInRF = kIsHalf && kMaxK <= kBlockSizeI;
    static constexpr bool kPreload = kIsHalf && Arch::kMinComputeCapability >= 80 && kOutputInRF;
    static constexpr int kBlockSizeJ = kPreload && kMaxK > 64 ? 128 : 64;
    using Kernel = AttentionBackwardKernel<
            Arch,
            Element,
            true,        // kIsAligned_
            false,       // kApplyDropout_
            kPreload,// kPreload_
            kBlockSizeI, // kBlockSizeI_,
            kBlockSizeJ, // kBlockSizeJ_,
            kMaxK        // kMaxK
        >;

  typename Kernel::Params p;
  {
    p.head_dim = head_dim;
    p.head_dim_value = head_dim_value;
    p.num_queries = seq_len;
    p.num_keys = seq_len_kv;
    p.num_heads = num_heads;
    if (causal)
      p.custom_mask_type = Kernel::CausalFromTopLeft;
    p.num_batches = batch_size;
    p.gB_strideM = seq_len_kv;
    p.scale = softmax_scale;

    p.query_ptr = static_cast<Element*>(query);
    p.q_strideH = head_dim;
    p.q_strideM = head_dim * num_heads;
    p.q_strideB = head_dim * num_heads * seq_len;
    p.key_ptr = static_cast<Element*>(key);
    p.k_strideH = head_dim;
    p.k_strideM = head_dim * num_heads;
    p.k_strideB = head_dim * num_heads * seq_len_kv;
    p.value_ptr = static_cast<Element*>(value);
    p.v_strideH = head_dim_value;
    p.v_strideM = head_dim_value * num_heads;
    p.v_strideB = head_dim_value * num_heads * seq_len_kv;
    p.bias_ptr = static_cast<Element*>(bias);
    p.bias_strideM = seq_len_kv;
    p.bias_strideH = seq_len_kv * seq_len;
    p.bias_strideB = seq_len_kv * seq_len * num_heads;
    p.logsumexp_ptr = static_cast<Kernel::lse_scalar_t*>(lse);
    p.lse_strideH = seq_len;
    p.lse_strideB = seq_len * num_heads;
    p.output_ptr = static_cast<Element*>(output);
    p.o_strideH = head_dim_value;
    auto o_strideM = head_dim_value * num_heads;
    if (o_strideM != p.o_strideM()) {
        std::cerr << "Invalid `o_strideM`: " << o_strideM << " - expected " << p.o_strideM();
        return 2;
    }
    p.o_strideB = head_dim_value * num_heads * seq_len;
    p.grad_output_ptr = static_cast<Element*>(grad_output);
    p.gO_strideH = head_dim_value;
    p.gO_strideM = head_dim_value * num_heads;
    p.gO_strideB = head_dim_value * num_heads * seq_len;
    p.delta_ptr = static_cast<typename Kernel::accum_t*>(delta);
    p.delta_strideH = seq_len;
    p.delta_strideB = seq_len * num_heads;

    // Allocate workspace
    if (p.workspace_size()) {
        cudaMalloc(&p.workspace, p.workspace_size());
    }
    p.grad_query_ptr = static_cast<Element*>(grad_q);
    p.grad_key_ptr = static_cast<Element*>(grad_k);
    p.grad_value_ptr = static_cast<Element*>(grad_v);
    p.grad_bias_ptr = static_cast<Element*>(grad_b);
    
    p.gQKV_strideM_multiplier = 1;
    p.gQ_strideH = head_dim;
    p.gQ_strideB = p.gQ_strideM() * seq_len;
    p.gK_strideH = head_dim;
    p.gK_strideB = p.gK_strideM() * seq_len_kv;
    p.gV_strideH = head_dim_value;
    p.gV_strideB = p.gV_strideM() * seq_len_kv;
    p.gB_strideH = seq_len_kv * seq_len;
    p.gB_strideB = seq_len_kv * seq_len * num_heads;
  }

  if (!Kernel::check_supported(p)) {
      std::cerr << "Kernel does not support these inputs" << std::endl;
      return -1;
    }
//   cudaDeviceSynchronize();
  auto kernel_fn = attention_kernel_backward_batched_impl<Kernel>;

  int smem_bytes = sizeof(typename Kernel::SharedStorage);
  if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
    cudaDeviceSynchronize();
    return 0;
}
