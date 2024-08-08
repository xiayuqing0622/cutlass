
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv_operation_3x.hpp"
#include "cutlass/conv/device/conv_universal_adapter.hpp"
#include "cutlass/conv/kernel/conv_universal.hpp"
#include "cutlass/conv/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////



// CUTLASS >= 3 convolution Fprop kernel instance "cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma"
using cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_64, cute::_64, cute::Shape<cute::_32>>,        // output cta tile shape
    cute::Shape<cute::_2, cute::_1, cute::_1>,                // cluster shape
    cutlass::epilogue::collective::EpilogueTileAuto,
    float,
    float,
    float, cutlass::layout::TensorNHWC, 128 / cute::sizeof_bits_v<float>,
    float, cutlass::layout::TensorNHWC, 128 / cute::sizeof_bits_v<float>,
    cutlass::epilogue::TmaWarpSpecialized
    // , class FusionOpOrCallbacks = cutlass::epilogue::fusion::LinearCombination<ElementD,ElementCompute>
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma_mainloop =
  typename cutlass::conv::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    cutlass::conv::Operator::kFprop,         // kFprop, kDgrad, or kWgrad
    float, cutlass::layout::TensorNHWC, 128 / cute::sizeof_bits_v<float>,
    float, cutlass::layout::TensorNHWC, 128 / cute::sizeof_bits_v<float>,
    float,
    cute::Shape<cute::_64, cute::_64, cute::Shape<cute::_32>>,        // mma tile shape
    cute::Shape<cute::_2, cute::_1, cute::_1>,         // cluster shape
    cutlass::conv::collective::StageCountAutoCarveout<sizeof(typename cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma_epilogue::SharedStorage)>,
    cutlass::conv::KernelImplicitTmaWarpSpecializedSm90
  >::CollectiveOp;

// Unit tests call this "ConvKernel".
// Conv operator cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma
using cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma_base = cutlass::conv::kernel::ConvUniversal<
    cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma_mainloop,
    cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma_epilogue,
    void
  >;

// Derived class
struct cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma :
  public cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



  using Operation_cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma = cutlass::conv::device::ConvUniversalAdapter<
    cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma>;

auto cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma_operation = new cutlass::library::ConvOperation3x<
      Operation_cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma
    >(
      "cutlass3x_sm90_tensorop_f3264x64x8fprop_f32nhwc_f32nhwc_f32_f32_f32_64x64x32_2x1x1_0_align16_warpspecialized_epi_tma"
);
