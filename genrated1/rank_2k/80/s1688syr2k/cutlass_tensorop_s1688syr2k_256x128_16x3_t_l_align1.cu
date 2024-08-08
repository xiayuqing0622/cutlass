#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "rank_2k_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Rank K operator cutlass_tensorop_s1688syr2k_256x128_16x3_t_l_align1
using Operation_cutlass_tensorop_s1688syr2k_256x128_16x3_t_l_align1 =
  typename cutlass::gemm::device::Rank2K<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor, cutlass::FillMode::kLower,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddFastF32
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_s1688syr2k_256x128_16x3_t_l_align1_operation = new Rank2KOperation<
    Operation_cutlass_tensorop_s1688syr2k_256x128_16x3_t_l_align1
>("cutlass_tensorop_s1688syr2k_256x128_16x3_t_l_align1");
