
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv2d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dFprop Fixed_channels kernel instance "cutlass_tensorop_h1688fprop_fixed_channels_128x64_32x2_nhwc_align4"
  using cutlass_tensorop_h1688fprop_fixed_channels_128x64_32x2_nhwc_align4_base =
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 16, 32 >,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      4,
      cutlass::half_t,
      cutlass::half_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kFixedChannels,
    cutlass::conv::StrideSupport::kStrided,
    4,
    4
  >::Kernel;

// Derived class
struct cutlass_tensorop_h1688fprop_fixed_channels_128x64_32x2_nhwc_align4 :
  public cutlass_tensorop_h1688fprop_fixed_channels_128x64_32x2_nhwc_align4_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



  using Operation_cutlass_tensorop_h1688fprop_fixed_channels_128x64_32x2_nhwc_align4 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_h1688fprop_fixed_channels_128x64_32x2_nhwc_align4>;

auto cutlass_tensorop_h1688fprop_fixed_channels_128x64_32x2_nhwc_align4_operation = new cutlass::library::Conv2dOperation<
      Operation_cutlass_tensorop_h1688fprop_fixed_channels_128x64_32x2_nhwc_align4
    >(
      "cutlass_tensorop_h1688fprop_fixed_channels_128x64_32x2_nhwc_align4"
);
