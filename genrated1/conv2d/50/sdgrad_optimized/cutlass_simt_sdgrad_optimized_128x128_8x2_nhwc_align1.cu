
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv2d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dDgrad Optimized kernel instance "cutlass_simt_sdgrad_optimized_128x128_8x2_nhwc_align1"
  using cutlass_simt_sdgrad_optimized_128x128_8x2_nhwc_align1_base =
  typename cutlass::conv::kernel::DefaultConv2dDgrad<
    float,
    cutlass::layout::TensorNHWC,
    float,
    cutlass::layout::TensorNHWC,
    float,
    cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8 >,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >,
    cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    1,
    1
  >::Kernel;

// Derived class
struct cutlass_simt_sdgrad_optimized_128x128_8x2_nhwc_align1 :
  public cutlass_simt_sdgrad_optimized_128x128_8x2_nhwc_align1_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



  using Operation_cutlass_simt_sdgrad_optimized_128x128_8x2_nhwc_align1 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_simt_sdgrad_optimized_128x128_8x2_nhwc_align1>;

auto cutlass_simt_sdgrad_optimized_128x128_8x2_nhwc_align1_operation = new cutlass::library::Conv2dOperation<
      Operation_cutlass_simt_sdgrad_optimized_128x128_8x2_nhwc_align1
    >(
      "cutlass_simt_sdgrad_optimized_128x128_8x2_nhwc_align1"
);
