#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "trmm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Trmm operator cutlass_tensorop_d1684trmm_128x128x16_1x1x1_3_nn_rs_l_un_align1
using Operation_cutlass_tensorop_d1684trmm_128x128x16_1x1x1_3_nn_rs_l_un_align1 =
  typename cutlass::gemm::device::Trmm<
    double, cutlass::layout::ColumnMajor,
    cutlass::SideMode::kRight, cutlass::FillMode::kLower, cutlass::DiagType::kUnit,
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::ColumnMajor,
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
      double,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAdd
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_d1684trmm_128x128x16_1x1x1_3_nn_rs_l_un_align1_operation = new TrmmOperation<
    Operation_cutlass_tensorop_d1684trmm_128x128x16_1x1x1_3_nn_rs_l_un_align1
>("cutlass_tensorop_d1684trmm_128x128x16_1x1x1_3_nn_rs_l_un_align1");
