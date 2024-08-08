
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv2d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dFprop Optimized kernel instance "cutlass_tensorop_i16832fprop_optimized_s8_256x128_64x3_nhwc_align16"
  using cutlass_tensorop_i16832fprop_optimized_s8_256x128_64x3_nhwc_align16_base =
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    int8_t,
    cutlass::layout::TensorNHWC,
    int8_t,
    cutlass::layout::TensorNHWC,
    int32_t,
    cutlass::layout::TensorNHWC,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64 >,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kUnity,
    16,
    16
  >::Kernel;

// Derived class
struct cutlass_tensorop_i16832fprop_optimized_s8_256x128_64x3_nhwc_align16 :
  public cutlass_tensorop_i16832fprop_optimized_s8_256x128_64x3_nhwc_align16_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



  using Operation_cutlass_tensorop_i16832fprop_optimized_s8_256x128_64x3_nhwc_align16 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_i16832fprop_optimized_s8_256x128_64x3_nhwc_align16>;

auto cutlass_tensorop_i16832fprop_optimized_s8_256x128_64x3_nhwc_align16_operation = new cutlass::library::Conv2dOperation<
      Operation_cutlass_tensorop_i16832fprop_optimized_s8_256x128_64x3_nhwc_align16
    >(
      "cutlass_tensorop_i16832fprop_optimized_s8_256x128_64x3_nhwc_align16"
);
