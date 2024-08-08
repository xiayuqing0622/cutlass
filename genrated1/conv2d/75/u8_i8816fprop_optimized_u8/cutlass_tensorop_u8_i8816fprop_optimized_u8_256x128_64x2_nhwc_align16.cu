
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv2d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dFprop Optimized kernel instance "cutlass_tensorop_u8_i8816fprop_optimized_u8_256x128_64x2_nhwc_align16"
  using cutlass_tensorop_u8_i8816fprop_optimized_u8_256x128_64x2_nhwc_align16_base =
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    uint8_t,
    cutlass::layout::TensorNHWC,
    uint8_t,
    cutlass::layout::TensorNHWC,
    uint8_t,
    cutlass::layout::TensorNHWC,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64 >,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      uint8_t,
      16,
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
struct cutlass_tensorop_u8_i8816fprop_optimized_u8_256x128_64x2_nhwc_align16 :
  public cutlass_tensorop_u8_i8816fprop_optimized_u8_256x128_64x2_nhwc_align16_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



  using Operation_cutlass_tensorop_u8_i8816fprop_optimized_u8_256x128_64x2_nhwc_align16 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_u8_i8816fprop_optimized_u8_256x128_64x2_nhwc_align16>;

auto cutlass_tensorop_u8_i8816fprop_optimized_u8_256x128_64x2_nhwc_align16_operation = new cutlass::library::Conv2dOperation<
      Operation_cutlass_tensorop_u8_i8816fprop_optimized_u8_256x128_64x2_nhwc_align16
    >(
      "cutlass_tensorop_u8_i8816fprop_optimized_u8_256x128_64x2_nhwc_align16"
);
