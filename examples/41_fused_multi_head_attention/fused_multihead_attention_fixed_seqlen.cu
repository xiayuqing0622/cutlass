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

using Arch = cutlass::arch::Sm70;
static constexpr int kMaxK = 128;

template <typename ArchTag, typename Element, int kMaxK>
struct DefaultKernel {
    // Some heuristics to select the best kernel (tested on Sm60, Sm70, Sm80)
    // NOTE: Requires quite a lot of shmem for Sm80+,
    // so might require tweaking those manually for Sm86/Sm89

  static constexpr int kQueriesPerBlock =  kMaxK > 64 ? 32 : 64;
  static constexpr int kKeysPerBlock =  kMaxK > 64 ? 128 : 64;
  static constexpr bool kSingleValueIteration =  kMaxK <= kKeysPerBlock ? true : false;

  using Attention = AttentionKernel<
    Element,      // scalar_t
    Arch,  // ArchTag
    true,                 // Memory is aligned
    kQueriesPerBlock,
    kKeysPerBlock,
    kSingleValueIteration,
    false,                // Supports dropout
    true                 // Supports bias
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
template <typename T> struct TypeName;
template <> struct TypeName<float> { static constexpr const char* Name = "f32"; };
template <> struct TypeName<cutlass::half_t> { static constexpr const char* Name = "f16"; };
template <> struct TypeName<cutlass::bfloat16_t> { static constexpr const char* Name = "b16"; };

void readExpect(std::string const& expected) {
    std::string read;
    std::cin >> read;
    if (read != expected) {
        std::cerr << "FATAL: Read '" << read << "' but expected '" << expected << "'" << std::endl;
        std::exit(1);
    }
}

/// Helpers to read from stdin
template <typename Element>
cutlass::HostTensor<Element, cutlass::layout::RowMajor> readTensorOnDevice(std::string const& expectedName) {
    readExpect("tensor_begin");
    readExpect(std::string(TypeName<Element>::Name) + ":" + expectedName);
    uint64_t len = 0;
    std::cin >> len;
    readExpect("file");
    std::string filename;
    std::cin >> filename;

    cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor({int64_t(1), int64_t(len / sizeof(Element))});
    uint8_t* data = (uint8_t*)tensor.host_data();

    std::fstream myFile(filename, std::ios::in | std::ios::binary );
    myFile.read((char*)data, len);
    readExpect("tensor_end");
    tensor.sync_device();
    return tensor;
}

int64_t readInt64(std::string const& expectedName) {
    readExpect(expectedName);
    int64_t s = 0;
    std::cin >> s;
    return s;
}

float readFloat(std::string const& expectedName) {
    readExpect(expectedName);
    float s = 0;
    std::cin >> s;
    return s;
}

// Writing
template <typename Element>
void writeTensor(std::string const& name, cutlass::HostTensor<Element, cutlass::layout::RowMajor>& tensor) {
    tensor.sync_host(); // device->host
    size_t u8len = tensor.size() * sizeof(Element);

    // Python is expected to provide a file name to write to
    readExpect("tmpfile");
    std::string tmpfile;
    std::cin >> tmpfile;

    uint8_t* data = (uint8_t*)tensor.host_data();
    std::fstream myFile(tmpfile, std::ios::out | std::ios::binary );
    myFile.write((char*)data, u8len);
    myFile.close();

    std::cout << "tensor_begin " << TypeName<Element>::Name << ":" << name << " ";
    std::cout << u8len << " file " << tmpfile << " tensor_end" << std::endl;
}

void writeInt64(std::string const& name, int64_t value) {
    std::cout << name << " " << value << std::endl;
}

void writeInt32(std::string const& name, int32_t value) {
    std::cout << name << " " << value << std::endl;
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element>
int runKernel() {
    using Kernel = typename DefaultKernel<Arch, Element, kMaxK>::Attention;

#define READ_I64(NAME) p.NAME = (decltype(p.NAME))readInt64(#NAME)
#define READ_TENSOR_AND_STRIDES_BMH(DT, NAME, NAME_XS) \
    auto storage##NAME = readTensorOnDevice<DT>(#NAME); \
    p.NAME##_ptr = storage##NAME.device_data(); \
    READ_I64(NAME_XS##_strideB); \
    READ_I64(NAME_XS##_strideM); \
    READ_I64(NAME_XS##_strideH);

#define CUDA_CHECK(FN) { \
    auto cudaError = FN; \
    if (cudaError != cudaSuccess) { \
        std::cerr << "FATAL: " #FN " failed: " << cudaGetErrorString(cudaError) << std::endl; \
        return -1; \
    } \
}

    typename Kernel::Params p;
    p.scale = readFloat("scale");
    READ_I64(head_dim);
    READ_I64(head_dim_value);
    READ_I64(num_queries);
    READ_I64(num_keys);
    READ_I64(num_heads);
    READ_I64(custom_mask_type);
    READ_I64(num_batches);
    int64_t repeat_count = readInt64("repeat_count");


    READ_TENSOR_AND_STRIDES_BMH(Element, query, q);
    READ_TENSOR_AND_STRIDES_BMH(Element, key, k);
    READ_TENSOR_AND_STRIDES_BMH(Element, value, v);

    auto bias = readTensorOnDevice<Element>("bias");
    p.attn_bias_ptr = bias.device_data();
    p.bias_strideB = READ_I64(bias_strideB);
    p.bias_strideH = READ_I64(bias_strideH);
    p.bias_strideM = READ_I64(bias_strideM);

    p.output_accum_ptr = nullptr;
    if (Kernel::kNeedsOutputAccumulatorBuffer) {
        cudaMalloc(&p.output_accum_ptr, p.num_batches * p.num_heads * p.num_queries * p.head_dim_value * sizeof(typename Kernel::output_accum_t));
      }
    // output

    // Allocate outputs in BMHK format

    p.o_strideH = p.head_dim_value;
    p.o_strideM = p.head_dim_value * p.num_heads;
    p.o_strideB = p.o_strideM * p.num_queries;
    p.lse_strideH = p.num_queries;
    p.lse_strideB = p.lse_strideH * p.num_heads;
    
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> output({1, p.o_strideB * p.num_batches});
    cutlass::HostTensor<typename Kernel::lse_scalar_t, cutlass::layout::RowMajor> lse({1, p.lse_strideB * p.num_batches});


    p.output_ptr = output.device_data();
    p.logsumexp_ptr = lse.device_data(); // Only needed for bw
    if (!Kernel::check_supported(p)) {
      std::cerr << "FATAL: Kernel does not support these inputs" << std::endl;
      return 2;
    }

    // Run kernel
    cudaDeviceSynchronize();
    auto kernel_fn = attention_kernel_batched_impl<Kernel>;
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes)));
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    // Write outputs
    std::cout << "OK ";
    writeTensor("output", output);
    writeInt64("o_strideB", p.o_strideB);
    writeInt64("o_strideM", p.o_strideM);
    writeInt64("o_strideH", p.o_strideH);
    writeTensor("lse", lse);
    writeInt64("lse_strideB", p.lse_strideB);
    writeInt64("lse_strideH", p.lse_strideH);

    // Timing
    cudaEvent_t events[2];
    for (auto & event : events) {
      CUDA_CHECK(cudaEventCreate(&event));
    }
    CUDA_CHECK(cudaEventRecord(events[0]));
    for (int i = 0; i < repeat_count; ++i) {
        kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
    }
    CUDA_CHECK(cudaEventRecord(events[1]));
    CUDA_CHECK(cudaEventSynchronize(events[1]));
    // Measure elapsed runtime
    float runtime_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));

    std::cout << "runtime_ms " << runtime_ms / float(repeat_count) << std::endl;
    return 0;
}

int main() {
    std::ios_base::sync_with_stdio(false);

    std::string dtype;
    std::cin >> dtype;
    std::cerr << "Running kernel with dtype: " << dtype << std::endl;
    if (dtype == "f16") {
        return runKernel<cutlass::half_t>();
    } else if (dtype == "b16") {
        return runKernel<cutlass::bfloat16_t>();
    } else if (dtype == "f32") {
        return runKernel<float>();
    } else {
        std::cerr << "FATAL: Unknown dtype: " << dtype << std::endl;
        return 3;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
