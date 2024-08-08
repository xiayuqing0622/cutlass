
/*
  Generated by rank_2k_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "rank_2k_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Rank K operator cutlass_tensorop_gz1684syr2k_64x64x8_1x1x1_3_n_u_align1
using Operation_cutlass_tensorop_gz1684syr2k_64x64x8_1x1x1_3_n_u_align1 =
  typename cutlass::gemm::device::Rank2K<
    cutlass::complex<double>, cutlass::layout::ColumnMajor,
    cutlass::complex<double>, cutlass::layout::ColumnMajor,
    cutlass::complex<double>, cutlass::layout::ColumnMajor, cutlass::FillMode::kUpper,
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
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    cutlass::BlasMode::kSymmetric
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_gz1684syr2k_64x64x8_1x1x1_3_n_u_align1(Manifest &manifest) {



  manifest.append(new Rank2KOperation<
    Operation_cutlass_tensorop_gz1684syr2k_64x64x8_1x1x1_3_n_u_align1
  >("cutlass_tensorop_gz1684syr2k_64x64x8_1x1x1_3_n_u_align1"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

