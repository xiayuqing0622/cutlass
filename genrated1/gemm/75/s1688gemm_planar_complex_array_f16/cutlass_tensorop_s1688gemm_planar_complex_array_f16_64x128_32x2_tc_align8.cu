
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "library_internal.h"
#include "gemm_operation.h"
#include "gemm_operation_3x.hpp"
#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Gemm operator cutlass_tensorop_s1688gemm_planar_complex_array_f16_64x128_32x2_tc_align8
  using Operation_cutlass_tensorop_s1688gemm_planar_complex_array_f16_64x128_32x2_tc_align8 = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kConjugate, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationPlanarComplex<
      float,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd
  >::GemmArrayKernel;

  struct cutlass_tensorop_s1688gemm_planar_complex_array_f16_64x128_32x2_tc_align8 : public Operation_cutlass_tensorop_s1688gemm_planar_complex_array_f16_64x128_32x2_tc_align8 { };


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_s1688gemm_planar_complex_array_f16_64x128_32x2_tc_align8_operation = new GemmPlanarComplexArrayOperation<
    cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_s1688gemm_planar_complex_array_f16_64x128_32x2_tc_align8>
>("cutlass_tensorop_s1688gemm_planar_complex_array_f16_64x128_32x2_tc_align8");
