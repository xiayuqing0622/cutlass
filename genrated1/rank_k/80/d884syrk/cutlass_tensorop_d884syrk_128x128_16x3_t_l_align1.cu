#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "rank_k_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Rank K operator cutlass_tensorop_d884syrk_128x128_16x3_t_l_align1
using Operation_cutlass_tensorop_d884syrk_128x128_16x3_t_l_align1 =
  typename cutlass::gemm::device::RankK<
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::ColumnMajor, cutlass::FillMode::kLower,
    double,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      double,
      1,
      double,
      double
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    false,
    cutlass::arch::OpMultiplyAdd
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_d884syrk_128x128_16x3_t_l_align1_operation = new RankKOperation<
    Operation_cutlass_tensorop_d884syrk_128x128_16x3_t_l_align1
>("cutlass_tensorop_d884syrk_128x128_16x3_t_l_align1");
