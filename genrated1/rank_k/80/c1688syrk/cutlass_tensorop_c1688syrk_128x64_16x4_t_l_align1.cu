#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "rank_k_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Rank K operator cutlass_tensorop_c1688syrk_128x64_16x4_t_l_align1
using Operation_cutlass_tensorop_c1688syrk_128x64_16x4_t_l_align1 =
  typename cutlass::gemm::device::RankK<
    cutlass::complex<float>, cutlass::layout::RowMajor,
    cutlass::complex<float>, cutlass::layout::ColumnMajor, cutlass::FillMode::kLower,
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
    false,
    cutlass::arch::OpMultiplyAddComplexFastF32,
    cutlass::ComplexTransform::kNone,
    cutlass::BlasMode::kSymmetric
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_c1688syrk_128x64_16x4_t_l_align1_operation = new RankKOperation<
    Operation_cutlass_tensorop_c1688syrk_128x64_16x4_t_l_align1
>("cutlass_tensorop_c1688syrk_128x64_16x4_t_l_align1");
