
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "library_internal.h"
#include "gemm_operation.h"
#include "gemm_operation_3x.hpp"
#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Gemm operator cutlass_tensorop_i16864spgemm_s8_128x64_128x3_tn_align16
  using Operation_cutlass_tensorop_i16864spgemm_s8_128x64_128x3_tn_align16 = cutlass::gemm::device::SparseGemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 128>,
    cutlass::gemm::GemmShape<64, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    16,
    16,
    false,
    cutlass::arch::OpMultiplyAddSaturate
    
  >;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_i16864spgemm_s8_128x64_128x3_tn_align16_operation = new GemmSparseOperation<Operation_cutlass_tensorop_i16864spgemm_s8_128x64_128x3_tn_align16>("cutlass_tensorop_i16864spgemm_s8_128x64_128x3_tn_align16");
