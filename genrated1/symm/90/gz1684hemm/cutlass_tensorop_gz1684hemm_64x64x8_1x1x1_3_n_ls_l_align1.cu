#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "symm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Symm operator cutlass_tensorop_gz1684hemm_64x64x8_1x1x1_3_n_ls_l_align1
using Operation_cutlass_tensorop_gz1684hemm_64x64x8_1x1x1_3_n_ls_l_align1 =
  typename cutlass::gemm::device::Symm<
    cutlass::complex<double>, cutlass::layout::ColumnMajor, cutlass::SideMode::kLeft, cutlass::FillMode::kLower,
    cutlass::complex<double>, cutlass::layout::ColumnMajor,
    cutlass::complex<double>, cutlass::layout::ColumnMajor,
    cutlass::complex<double>,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<16, 32, 8>,
    cutlass::gemm::GemmShape<16, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::complex<double>,
      1,
      cutlass::complex<double>,
      cutlass::complex<double>
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddGaussianComplex,
    cutlass::BlasMode::kHermitian
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_gz1684hemm_64x64x8_1x1x1_3_n_ls_l_align1_operation = new SymmOperation<
    Operation_cutlass_tensorop_gz1684hemm_64x64x8_1x1x1_3_n_ls_l_align1
>("cutlass_tensorop_gz1684hemm_64x64x8_1x1x1_3_n_ls_l_align1");
