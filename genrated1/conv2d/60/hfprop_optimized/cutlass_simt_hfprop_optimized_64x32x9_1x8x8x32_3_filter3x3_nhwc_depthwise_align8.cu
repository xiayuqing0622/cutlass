
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv2d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dFprop Optimized kernel instance "cutlass_simt_hfprop_optimized_64x32x9_1x8x8x32_3_filter3x3_nhwc_depthwise_align8"
  using cutlass_simt_hfprop_optimized_64x32x9_1x8x8x32_3_filter3x3_nhwc_depthwise_align8_base =
  typename cutlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm60,
    cutlass::gemm::GemmShape<64, 32, 9>,
    cutlass::conv::TensorNHWCShape<1, 8, 8, 32>,
    cutlass::MatrixShape<3, 3>,
    cutlass::gemm::GemmShape<16, 32, 9>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      cutlass::half_t,
      cutlass::half_t,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >,

    cutlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
          1,
          1,
          8,
          8>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    cutlass::MatrixShape<-1, -1>,
    cutlass::MatrixShape<-1, -1>
  >::Kernel;

// Derived class
struct cutlass_simt_hfprop_optimized_64x32x9_1x8x8x32_3_filter3x3_nhwc_depthwise_align8 :
  public cutlass_simt_hfprop_optimized_64x32x9_1x8x8x32_3_filter3x3_nhwc_depthwise_align8_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



  using Operation_cutlass_simt_hfprop_optimized_64x32x9_1x8x8x32_3_filter3x3_nhwc_depthwise_align8 = cutlass::conv::device::DirectConvolution<
    cutlass_simt_hfprop_optimized_64x32x9_1x8x8x32_3_filter3x3_nhwc_depthwise_align8>;

auto cutlass_simt_hfprop_optimized_64x32x9_1x8x8x32_3_filter3x3_nhwc_depthwise_align8_operation = new cutlass::library::DirectConv2dOperation<
      Operation_cutlass_simt_hfprop_optimized_64x32x9_1x8x8x32_3_filter3x3_nhwc_depthwise_align8
    >(
      "cutlass_simt_hfprop_optimized_64x32x9_1x8x8x32_3_filter3x3_nhwc_depthwise_align8"
);
