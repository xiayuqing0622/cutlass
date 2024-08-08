#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "trmm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Trmm operator cutlass_tensorop_gz1684trmm_64x64x8_1x1x1_3_cn_ls_l_un_align1
using Operation_cutlass_tensorop_gz1684trmm_64x64x8_1x1x1_3_cn_ls_l_un_align1 =
  typename cutlass::gemm::device::Trmm<
    cutlass::complex<double>, cutlass::layout::ColumnMajor,
    cutlass::SideMode::kLeft, cutlass::FillMode::kLower, cutlass::DiagType::kUnit,
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
      cutlass::complex<double>,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddGaussianComplex,
    cutlass::ComplexTransform::kConjugate
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_gz1684trmm_64x64x8_1x1x1_3_cn_ls_l_un_align1_operation = new TrmmOperation<
    Operation_cutlass_tensorop_gz1684trmm_64x64x8_1x1x1_3_cn_ls_l_un_align1
>("cutlass_tensorop_gz1684trmm_64x64x8_1x1x1_3_cn_ls_l_un_align1");
