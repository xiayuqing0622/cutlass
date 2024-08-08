
/*
  Generated by conv2d_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv2d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dFprop Optimized kernel instance "cutlass_tensorop_s8_i8816fprop_optimized_s8_256x128_64x2_nc32hw32_align16"
  using cutlass_tensorop_s8_i8816fprop_optimized_s8_256x128_64x2_nc32hw32_align16_base =
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    int8_t,
    cutlass::layout::TensorNCxHWx<32>,
    int8_t,
    cutlass::layout::TensorCxRSKx<32>,
    int8_t,
    cutlass::layout::TensorNCxHWx<32>,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64 >,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      int8_t,
      8,
      int32_t,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kUnity,
    16,
    16
  >::Kernel;

// Derived class
struct cutlass_tensorop_s8_i8816fprop_optimized_s8_256x128_64x2_nc32hw32_align16 :
  public cutlass_tensorop_s8_i8816fprop_optimized_s8_256x128_64x2_nc32hw32_align16_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



namespace cutlass {
namespace library {

// Initialize all instances
void initialize_cutlass_tensorop_s8_i8816fprop_optimized_s8_256x128_64x2_nc32hw32_align16(Manifest &manifest) {

  using Operation_cutlass_tensorop_s8_i8816fprop_optimized_s8_256x128_64x2_nc32hw32_align16 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_s8_i8816fprop_optimized_s8_256x128_64x2_nc32hw32_align16>;

  manifest.append(new cutlass::library::Conv2dOperation<
      Operation_cutlass_tensorop_s8_i8816fprop_optimized_s8_256x128_64x2_nc32hw32_align16
    >(
      "cutlass_tensorop_s8_i8816fprop_optimized_s8_256x128_64x2_nc32hw32_align16"
    ));

}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

