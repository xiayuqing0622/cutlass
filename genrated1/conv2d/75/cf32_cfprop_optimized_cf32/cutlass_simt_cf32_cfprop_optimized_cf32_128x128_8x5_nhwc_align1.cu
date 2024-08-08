
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv2d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dFprop Optimized kernel instance "cutlass_simt_cf32_cfprop_optimized_cf32_128x128_8x5_nhwc_align1"
  using cutlass_simt_cf32_cfprop_optimized_cf32_128x128_8x5_nhwc_align1_base =
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::complex<float>,
    cutlass::layout::TensorNHWC,
    cutlass::complex<float>,
    cutlass::layout::TensorNHWC,
    cutlass::complex<float>,
    cutlass::layout::TensorNHWC,
    cutlass::complex<float>,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8 >,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::complex<float>,
      1,
      cutlass::complex<float>,
      cutlass::complex<float>
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    5,
    cutlass::arch::OpMultiplyAddComplex,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kUnity,
    1,
    1
  >::Kernel;

// Derived class
struct cutlass_simt_cf32_cfprop_optimized_cf32_128x128_8x5_nhwc_align1 :
  public cutlass_simt_cf32_cfprop_optimized_cf32_128x128_8x5_nhwc_align1_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



  using Operation_cutlass_simt_cf32_cfprop_optimized_cf32_128x128_8x5_nhwc_align1 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_simt_cf32_cfprop_optimized_cf32_128x128_8x5_nhwc_align1>;

auto cutlass_simt_cf32_cfprop_optimized_cf32_128x128_8x5_nhwc_align1_operation = new cutlass::library::Conv2dOperation<
      Operation_cutlass_simt_cf32_cfprop_optimized_cf32_128x128_8x5_nhwc_align1
    >(
      "cutlass_simt_cf32_cfprop_optimized_cf32_128x128_8x5_nhwc_align1"
);
