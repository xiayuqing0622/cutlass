
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv2d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dFprop Optimized kernel instance "cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nhwc_align32"
  using cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nhwc_align32_base =
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::uint4b_t,
    cutlass::layout::TensorNHWC,
    cutlass::uint4b_t,
    cutlass::layout::TensorNHWC,
    cutlass::uint4b_t,
    cutlass::layout::TensorNHWC,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128 >,
    cutlass::gemm::GemmShape<16, 8, 64>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      cutlass::uint4b_t,
      16,
      int32_t,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kUnity,
    32,
    32
  >::Kernel;

// Derived class
struct cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nhwc_align32 :
  public cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nhwc_align32_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



  using Operation_cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nhwc_align32 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nhwc_align32>;

auto cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nhwc_align32_operation = new cutlass::library::Conv2dOperation<
      Operation_cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nhwc_align32
    >(
      "cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nhwc_align32"
);
