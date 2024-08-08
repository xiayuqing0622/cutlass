
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "library_internal.h"
#include "gemm_operation.h"
#include "gemm_operation_3x.hpp"
#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Gemm operator cutlass_tensorop_s4_i168128spgemm_s4_64x64_256x4_tn_align32
  using Operation_cutlass_tensorop_s4_i168128spgemm_s4_64x64_256x4_tn_align32 = cutlass::gemm::device::SparseGemm<
    cutlass::int4b_t, cutlass::layout::RowMajor,
    cutlass::int4b_t, cutlass::layout::ColumnMajor,
    cutlass::int4b_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<32, 32, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      cutlass::int4b_t,
      8,
      int32_t,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    32,
    32,
    false,
    cutlass::arch::OpMultiplyAddSaturate
    
  >;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_s4_i168128spgemm_s4_64x64_256x4_tn_align32_operation = new GemmSparseOperation<Operation_cutlass_tensorop_s4_i168128spgemm_s4_64x64_256x4_tn_align32>("cutlass_tensorop_s4_i168128spgemm_s4_64x64_256x4_tn_align32");
