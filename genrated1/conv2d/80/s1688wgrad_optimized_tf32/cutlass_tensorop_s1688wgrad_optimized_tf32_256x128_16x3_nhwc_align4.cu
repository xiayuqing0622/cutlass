
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv2d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dWgrad Optimized kernel instance "cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4"
  using cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4_base =
  typename cutlass::conv::kernel::DefaultConv2dWgrad<
    cutlass::tfloat32_t,
    cutlass::layout::TensorNHWC,
    cutlass::tfloat32_t,
    cutlass::layout::TensorNHWC,
    float,
    cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16 >,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    4,
    4
  >::Kernel;

// Derived class
struct cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4 :
  public cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



  using Operation_cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4>;

auto cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4_operation = new cutlass::library::Conv2dOperation<
      Operation_cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4
    >(
      "cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4"
);
