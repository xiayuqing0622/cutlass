#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "trmm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Trmm operator cutlass_tensorop_d884trmm_128x128_16x3_nn_ls_u_nu_align1
using Operation_cutlass_tensorop_d884trmm_128x128_16x3_nn_ls_u_nu_align1 =
  typename cutlass::gemm::device::Trmm<
    double, cutlass::layout::ColumnMajor,
    cutlass::SideMode::kLeft, cutlass::FillMode::kUpper, cutlass::DiagType::kNonUnit,
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::ColumnMajor,
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

auto cutlass_tensorop_d884trmm_128x128_16x3_nn_ls_u_nu_align1_operation = new TrmmOperation<
    Operation_cutlass_tensorop_d884trmm_128x128_16x3_nn_ls_u_nu_align1
>("cutlass_tensorop_d884trmm_128x128_16x3_nn_ls_u_nu_align1");
