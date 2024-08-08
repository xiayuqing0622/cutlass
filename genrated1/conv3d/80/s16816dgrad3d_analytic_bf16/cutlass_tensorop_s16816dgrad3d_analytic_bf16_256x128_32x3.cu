
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv3d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv3dDgrad Analytic kernel instance "cutlass_tensorop_s16816dgrad3d_analytic_bf16_256x128_32x3"
  using cutlass_tensorop_s16816dgrad3d_analytic_bf16_256x128_32x3_base =
  typename cutlass::conv::kernel::DefaultConv3dDgrad<
    cutlass::bfloat16_t,
    cutlass::layout::TensorNDHWC,
    cutlass::bfloat16_t,
    cutlass::layout::TensorNDHWC,
    float,
    cutlass::layout::TensorNDHWC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32 >,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic,
    cutlass::conv::StrideSupport::kStrided
  >::Kernel;

// Derived class
struct cutlass_tensorop_s16816dgrad3d_analytic_bf16_256x128_32x3 :
  public cutlass_tensorop_s16816dgrad3d_analytic_bf16_256x128_32x3_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



  using Operation_cutlass_tensorop_s16816dgrad3d_analytic_bf16_256x128_32x3 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_s16816dgrad3d_analytic_bf16_256x128_32x3>;

auto cutlass_tensorop_s16816dgrad3d_analytic_bf16_256x128_32x3_operation = new cutlass::library::Conv3dOperation<
      Operation_cutlass_tensorop_s16816dgrad3d_analytic_bf16_256x128_32x3
    >(
      "cutlass_tensorop_s16816dgrad3d_analytic_bf16_256x128_32x3"
);
