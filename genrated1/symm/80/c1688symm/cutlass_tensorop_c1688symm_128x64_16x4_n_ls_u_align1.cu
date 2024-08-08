#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "symm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Symm operator cutlass_tensorop_c1688symm_128x64_16x4_n_ls_u_align1
using Operation_cutlass_tensorop_c1688symm_128x64_16x4_n_ls_u_align1 =
  typename cutlass::gemm::device::Symm<
    cutlass::complex<float>, cutlass::layout::ColumnMajor, cutlass::SideMode::kLeft, cutlass::FillMode::kUpper,
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
      cutlass::complex<float>
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddComplexFastF32,
    cutlass::BlasMode::kSymmetric
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_c1688symm_128x64_16x4_n_ls_u_align1_operation = new SymmOperation<
    Operation_cutlass_tensorop_c1688symm_128x64_16x4_n_ls_u_align1
>("cutlass_tensorop_c1688symm_128x64_16x4_n_ls_u_align1");
