
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv2d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dWgrad Optimized kernel instance "cutlass_tensorop_h16816wgrad_optimized_256x128_32x3_nhwc_align8"
  using cutlass_tensorop_h16816wgrad_optimized_256x128_32x3_nhwc_align8_base =
  typename cutlass::conv::kernel::DefaultConv2dWgrad<
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32 >,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      cutlass::half_t,
      cutlass::half_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    8,
    8
  >::Kernel;

// Derived class
struct cutlass_tensorop_h16816wgrad_optimized_256x128_32x3_nhwc_align8 :
  public cutlass_tensorop_h16816wgrad_optimized_256x128_32x3_nhwc_align8_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



  using Operation_cutlass_tensorop_h16816wgrad_optimized_256x128_32x3_nhwc_align8 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_h16816wgrad_optimized_256x128_32x3_nhwc_align8>;

auto cutlass_tensorop_h16816wgrad_optimized_256x128_32x3_nhwc_align8_operation = new cutlass::library::Conv2dOperation<
      Operation_cutlass_tensorop_h16816wgrad_optimized_256x128_32x3_nhwc_align8
    >(
      "cutlass_tensorop_h16816wgrad_optimized_256x128_32x3_nhwc_align8"
);
