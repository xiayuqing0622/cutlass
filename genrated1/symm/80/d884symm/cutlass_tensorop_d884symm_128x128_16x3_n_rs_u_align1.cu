#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "symm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Symm operator cutlass_tensorop_d884symm_128x128_16x3_n_rs_u_align1
using Operation_cutlass_tensorop_d884symm_128x128_16x3_n_rs_u_align1 =
  typename cutlass::gemm::device::Symm<
    double, cutlass::layout::ColumnMajor, cutlass::SideMode::kRight, cutlass::FillMode::kUpper,
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

auto cutlass_tensorop_d884symm_128x128_16x3_n_rs_u_align1_operation = new SymmOperation<
    Operation_cutlass_tensorop_d884symm_128x128_16x3_n_rs_u_align1
>("cutlass_tensorop_d884symm_128x128_16x3_n_rs_u_align1");
