#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "rank_2k_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Rank K operator cutlass_tensorop_d1684syr2k_128x128x16_1x1x1_3_t_u_align1
using Operation_cutlass_tensorop_d1684syr2k_128x128x16_1x1x1_3_t_u_align1 =
  typename cutlass::gemm::device::Rank2K<
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::ColumnMajor, cutlass::FillMode::kUpper,
    double,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      double,
      1,
      double,
      double
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAdd
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_d1684syr2k_128x128x16_1x1x1_3_t_u_align1_operation = new Rank2KOperation<
    Operation_cutlass_tensorop_d1684syr2k_128x128x16_1x1x1_3_t_u_align1
>("cutlass_tensorop_d1684syr2k_128x128x16_1x1x1_3_t_u_align1");
