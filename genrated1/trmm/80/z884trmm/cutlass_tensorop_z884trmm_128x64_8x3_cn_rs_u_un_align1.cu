#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "trmm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Trmm operator cutlass_tensorop_z884trmm_128x64_8x3_cn_rs_u_un_align1
using Operation_cutlass_tensorop_z884trmm_128x64_8x3_cn_rs_u_un_align1 =
  typename cutlass::gemm::device::Trmm<
    cutlass::complex<double>, cutlass::layout::ColumnMajor,
    cutlass::SideMode::kRight, cutlass::FillMode::kUpper, cutlass::DiagType::kUnit,
    cutlass::complex<double>, cutlass::layout::ColumnMajor,
    cutlass::complex<double>, cutlass::layout::ColumnMajor,
    cutlass::complex<double>,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<8, 8, 4>,
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
    cutlass::arch::OpMultiplyAddComplex,
    cutlass::ComplexTransform::kConjugate
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_z884trmm_128x64_8x3_cn_rs_u_un_align1_operation = new TrmmOperation<
    Operation_cutlass_tensorop_z884trmm_128x64_8x3_cn_rs_u_un_align1
>("cutlass_tensorop_z884trmm_128x64_8x3_cn_rs_u_un_align1");
