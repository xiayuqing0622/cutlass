#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "symm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Symm operator cutlass_tensorop_d1684symm_128x128x16_1x1x1_3_n_ls_u_align1
using Operation_cutlass_tensorop_d1684symm_128x128x16_1x1x1_3_n_ls_u_align1 =
  typename cutlass::gemm::device::Symm<
    double, cutlass::layout::ColumnMajor, cutlass::SideMode::kLeft, cutlass::FillMode::kUpper,
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

auto cutlass_tensorop_d1684symm_128x128x16_1x1x1_3_n_ls_u_align1_operation = new SymmOperation<
    Operation_cutlass_tensorop_d1684symm_128x128x16_1x1x1_3_n_ls_u_align1
>("cutlass_tensorop_d1684symm_128x128x16_1x1x1_3_n_ls_u_align1");
