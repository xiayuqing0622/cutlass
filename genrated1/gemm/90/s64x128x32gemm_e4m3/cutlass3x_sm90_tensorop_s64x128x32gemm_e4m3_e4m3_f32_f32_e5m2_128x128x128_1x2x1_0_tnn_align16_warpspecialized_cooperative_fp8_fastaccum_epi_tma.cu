
/*
  Generated by gemm_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "library_internal.h"
#include "gemm_operation.h"
#include "gemm_operation_3x.hpp"
#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////



using cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_f32_e5m2_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_1, cute::_2, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    float, cutlass::layout::ColumnMajor, 1,
    cutlass::float_e5m2_t, cutlass::layout::ColumnMajor, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    
    cutlass::epilogue::fusion::LinearCombination<
      cutlass::float_e5m2_t,
      float,
      float,
      float
    >

  >::CollectiveOp;

using cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_f32_e5m2_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_1, cute::_2, cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_f32_e5m2_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma_epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_f32_e5m2_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma
using cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_f32_e5m2_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_f32_e5m2_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma_mainloop,
    cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_f32_e5m2_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma_epilogue,
    cutlass::gemm::PersistentScheduler>;

// Define named type
struct cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_f32_e5m2_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma :
  public cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_f32_e5m2_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_f32_e5m2_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma_operation =
new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_f32_e5m2_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma");
