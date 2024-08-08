
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "library_internal.h"
#include "gemm_operation.h"
#include "gemm_operation_3x.hpp"
#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Gemm operator cutlass_tensorop_s16816tf32spgemm_128x64_32x3_tn_align4
  using Operation_cutlass_tensorop_s16816tf32spgemm_128x64_32x3_tn_align4 = cutlass::gemm::device::SparseGemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    false,
    cutlass::arch::OpMultiplyAdd
    
  >;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_s16816tf32spgemm_128x64_32x3_tn_align4_operation = new GemmSparseOperation<Operation_cutlass_tensorop_s16816tf32spgemm_128x64_32x3_tn_align4>("cutlass_tensorop_s16816tf32spgemm_128x64_32x3_tn_align4");
