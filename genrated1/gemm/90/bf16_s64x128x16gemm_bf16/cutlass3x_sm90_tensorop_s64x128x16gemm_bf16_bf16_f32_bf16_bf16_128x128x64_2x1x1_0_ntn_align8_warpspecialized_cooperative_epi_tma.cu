
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



using cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x128x64_2x1x1_0_ntn_align8_warpspecialized_cooperative_epi_tma_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    
    cutlass::epilogue::fusion::LinearCombination<
      cutlass::bfloat16_t,
      float,
      cutlass::bfloat16_t,
      float
    >

  >::CollectiveOp;

using cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x128x64_2x1x1_0_ntn_align8_warpspecialized_cooperative_epi_tma_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x128x64_2x1x1_0_ntn_align8_warpspecialized_cooperative_epi_tma_epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x128x64_2x1x1_0_ntn_align8_warpspecialized_cooperative_epi_tma
using cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x128x64_2x1x1_0_ntn_align8_warpspecialized_cooperative_epi_tma_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x128x64_2x1x1_0_ntn_align8_warpspecialized_cooperative_epi_tma_mainloop,
    cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x128x64_2x1x1_0_ntn_align8_warpspecialized_cooperative_epi_tma_epilogue,
    cutlass::gemm::PersistentScheduler>;

// Define named type
struct cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x128x64_2x1x1_0_ntn_align8_warpspecialized_cooperative_epi_tma :
  public cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x128x64_2x1x1_0_ntn_align8_warpspecialized_cooperative_epi_tma_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x128x64_2x1x1_0_ntn_align8_warpspecialized_cooperative_epi_tma_operation =
new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x128x64_2x1x1_0_ntn_align8_warpspecialized_cooperative_epi_tma");
