#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "trmm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Trmm operator cutlass_tensorop_c1688trmm_128x64_16x4_cn_rs_l_un_align1
using Operation_cutlass_tensorop_c1688trmm_128x64_16x4_cn_rs_l_un_align1 =
  typename cutlass::gemm::device::Trmm<
    cutlass::complex<float>, cutlass::layout::ColumnMajor,
    cutlass::SideMode::kRight, cutlass::FillMode::kLower, cutlass::DiagType::kUnit,
    cutlass::complex<float>, cutlass::layout::ColumnMajor,
    cutlass::complex<float>, cutlass::layout::ColumnMajor,
    cutlass::complex<float>,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::complex<float>,
      1,
      cutlass::complex<float>,
      cutlass::complex<float>,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddComplexFastF32,
    cutlass::ComplexTransform::kConjugate
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_c1688trmm_128x64_16x4_cn_rs_l_un_align1_operation = new TrmmOperation<
    Operation_cutlass_tensorop_c1688trmm_128x64_16x4_cn_rs_l_un_align1
>("cutlass_tensorop_c1688trmm_128x64_16x4_cn_rs_l_un_align1");
