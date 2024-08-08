#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "rank_2k_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Rank K operator cutlass_tensorop_c1688her2k_128x64_16x4_h_u_align1
using Operation_cutlass_tensorop_c1688her2k_128x64_16x4_h_u_align1 =
  typename cutlass::gemm::device::Rank2K<
    cutlass::complex<float>, cutlass::layout::RowMajor,
    cutlass::complex<float>, cutlass::layout::RowMajor,
    cutlass::complex<float>, cutlass::layout::ColumnMajor, cutlass::FillMode::kUpper,
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
      cutlass::complex<float>
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddComplexFastF32,
    cutlass::ComplexTransform::kConjugate,
    cutlass::ComplexTransform::kConjugate,
    cutlass::BlasMode::kHermitian
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_c1688her2k_128x64_16x4_h_u_align1_operation = new Rank2KOperation<
    Operation_cutlass_tensorop_c1688her2k_128x64_16x4_h_u_align1
>("cutlass_tensorop_c1688her2k_128x64_16x4_h_u_align1");
