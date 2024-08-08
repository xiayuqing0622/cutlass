
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "library_internal.h"
#include "gemm_operation.h"
#include "gemm_operation_3x.hpp"
#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Gemm operator cutlass_tensorop_bf16_s16832spgemm_bf16_64x128_64x6_nn_align8
  using Operation_cutlass_tensorop_bf16_s16832spgemm_bf16_64x128_64x6_nn_align8 = cutlass::gemm::device::SparseGemm<
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::bfloat16_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    6,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAdd
    
  >;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_bf16_s16832spgemm_bf16_64x128_64x6_nn_align8_operation = new GemmSparseOperation<Operation_cutlass_tensorop_bf16_s16832spgemm_bf16_64x128_64x6_nn_align8>("cutlass_tensorop_bf16_s16832spgemm_bf16_64x128_64x6_nn_align8");
