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

/*! \file
    \brief CUTLASS Attention Example.

    This workload computes a fused multi head attention.
    Because it keeps the attention matrix in shared memory, it's both faster and
    uses less global memory.

    This is based on `"Self-Attention Does Not Need O(n^2) Memory" <http://arxiv.org/abs/2112.05682>`_,
    and very similar to `"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" <https://arxiv.org/abs/2205.14135>`_.

    Algorithm:
      In short, we can compute the output incrementally in blocks of size B,
      we just need to divide the final result by the sum of all coefficients in
      the softmax (which we compute incrementally) with the following pseudo-code:

      ```
      s_prime = torch.zeros([num_queries, B])
      O = torch.zeros([num_queries, head_size_v])
      for i in range(0, K.shape[0], B):
        si = exp((Q . K[i * B:(i+1) * B].t) * scale)
        sum_coefs += attn_unscaled.sum(-1)
        O  += si . V[i * B:(i+1) * B]
      O = O / s_prime
      ```

      In practice, and for numerical stability reasons,
      we also substract the maximum so far (`mi`) before doing
      the exponential. When we encounter new keys, the maximum
      used to compute O so far (`m_prime`) can differ from the
      current maximum, so we update O before accumulating with

      ```
      O       = O * exp(m_prime - mi)
      m_prime = mi
      ```

    Implementation details:
      - `si` is stored in shared memory between the 2 back to back gemms
      - we keep and accumulate the output
      directly in registers if we can (`head_size_v <= 128`).
      Otherwise, we store it & accumulate in global memory (slower)
      - blocks are parallelized across the batch dimension, the number
      of heads, and the query sequence size


    Examples:

      # Run an attention example with default setup
      $ ./examples/41_fused_multi_head_attention/41_fused_multi_head_attention_fixed_seqlen

      # Run an attention example with custom setup
      $ ./examples/41_fused_multi_head_attention/41_fused_multi_head_attention_fixed_seqlen --head_number=2 --batch_size=3 --head_size=32 --head_size_v=64 --seq_length=512 --seq_length_kv=1024 --causal=true

      Acknowledgement: Fixed-sequence-length FMHA code was upstreamed by Meta xFormers (https://github.com/facebookresearch/xformers).
*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <iostream>
#include <fstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/fast_math.h"
#include "kernel_forward.h"

#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int kMaxK = 64;

extern "C" int flash_attn_fwd(void* output,
                   void* query,
                   void* key,
                   void* value,
                   void* bias,
                   void* lse,
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
  static constexpr int kQueriesPerBlock =  kMaxK > 64 ? 32 : 64;
  static constexpr int kKeysPerBlock =  kMaxK > 64 ? 128 : 64;
  static constexpr bool kSingleValueIteration =  kMaxK <= kKeysPerBlock ? true : false;
  using Kernel = AttentionKernel<
    Element, // scalar_t,
    Arch,  // ArchTag
    true,                 // Memory is aligned
    kQueriesPerBlock,
    kKeysPerBlock,
    kSingleValueIteration,
    false,                // Supports dropout
    true                 // Supports bias
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

    p.scale = softmax_scale;

    p.query_ptr = static_cast<Element*>(query);
    p.q_strideH = p.head_dim;
    p.q_strideM = p.q_strideH * p.num_heads;
    p.q_strideB = p.q_strideM * p.num_queries;
    p.key_ptr = static_cast<Element*>(key);
    p.k_strideH = p.head_dim;
    p.k_strideM = p.k_strideH * p.num_heads;
    p.k_strideB = p.k_strideM * p.num_keys;
    p.value_ptr = static_cast<Element*>(value);
    p.v_strideH = p.head_dim_value;
    p.v_strideM = p.v_strideH * p.num_heads;
    p.v_strideB = p.v_strideM * p.num_keys;
    p.attn_bias_ptr = static_cast<Element*>(bias);
    p.bias_strideM = p.num_keys;
    p.bias_strideH = p.bias_strideM * p.num_queries;
    p.bias_strideB = p.bias_strideH * p.num_heads;

    p.output_accum_ptr = nullptr;
    if (Kernel::kNeedsOutputAccumulatorBuffer) {
        cudaMalloc(&p.output_accum_ptr, p.num_batches * p.num_heads * p.num_queries * p.head_dim_value * sizeof(typename Kernel::output_accum_t));
      }
    p.output_ptr = static_cast<Element*>(output);
    p.logsumexp_ptr = static_cast<Kernel::lse_scalar_t*>(lse);// Only needed for bw
    p.o_strideH = p.head_dim_value;
    p.o_strideM = p.head_dim_value * p.num_heads;
    p.o_strideB = p.o_strideM * p.num_queries;
    p.lse_strideH = p.num_queries;
    p.lse_strideB = p.lse_strideH * p.num_heads;
  }
  if (!Kernel::check_supported(p)) {
      std::cerr << "Kernel does not support these inputs" << std::endl;
      return -1;
    }
  // launch kernel
  auto kernel_fn = attention_kernel_batched_impl<Kernel>;
  int smem_bytes = sizeof(typename Kernel::SharedStorage);
  if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
    cudaDeviceSynchronize();
    return 0;
}
