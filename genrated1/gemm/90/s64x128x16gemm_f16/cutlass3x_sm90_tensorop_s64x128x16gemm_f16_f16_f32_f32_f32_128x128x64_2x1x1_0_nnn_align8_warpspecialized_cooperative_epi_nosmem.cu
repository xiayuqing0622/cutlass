
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



using cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f32_f32_128x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    float, cutlass::layout::ColumnMajor, 4,
    float, cutlass::layout::ColumnMajor, 4,
    cutlass::epilogue::NoSmemWarpSpecialized,
    
    cutlass::epilogue::fusion::LinearCombination<
      float,
      float,
      float,
      float
    >

  >::CollectiveOp;

using cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f32_f32_128x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f32_f32_128x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem_epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f32_f32_128x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem
using cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f32_f32_128x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f32_f32_128x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem_mainloop,
    cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f32_f32_128x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem_epilogue,
    cutlass::gemm::PersistentScheduler>;

// Define named type
struct cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f32_f32_128x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem :
  public cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f32_f32_128x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f32_f32_128x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem_operation =
new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f32_f32_128x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem");
