
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



using cutlass3x_sm90_tensorop_i64x128x32gemm_u8_u8_s32_s32_s32_128x128x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_epi_nosmem_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    int32_t, int32_t,
    int32_t, cutlass::layout::ColumnMajor, 4,
    int32_t, cutlass::layout::ColumnMajor, 4,
    cutlass::epilogue::NoSmemWarpSpecialized,
    
    cutlass::epilogue::fusion::LinearCombination<
      int32_t,
      int32_t,
      int32_t,
      int32_t
    >

  >::CollectiveOp;

using cutlass3x_sm90_tensorop_i64x128x32gemm_u8_u8_s32_s32_s32_128x128x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_epi_nosmem_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    uint8_t, cutlass::layout::RowMajor, 16,
    uint8_t, cutlass::layout::ColumnMajor, 16,
    int32_t,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_i64x128x32gemm_u8_u8_s32_s32_s32_128x128x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_epi_nosmem_epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_i64x128x32gemm_u8_u8_s32_s32_s32_128x128x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_epi_nosmem
using cutlass3x_sm90_tensorop_i64x128x32gemm_u8_u8_s32_s32_s32_128x128x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_epi_nosmem_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_i64x128x32gemm_u8_u8_s32_s32_s32_128x128x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_epi_nosmem_mainloop,
    cutlass3x_sm90_tensorop_i64x128x32gemm_u8_u8_s32_s32_s32_128x128x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_epi_nosmem_epilogue,
    cutlass::gemm::PersistentScheduler>;

// Define named type
struct cutlass3x_sm90_tensorop_i64x128x32gemm_u8_u8_s32_s32_s32_128x128x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_epi_nosmem :
  public cutlass3x_sm90_tensorop_i64x128x32gemm_u8_u8_s32_s32_s32_128x128x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_epi_nosmem_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass3x_sm90_tensorop_i64x128x32gemm_u8_u8_s32_s32_s32_128x128x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_epi_nosmem_operation =
new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_i64x128x32gemm_u8_u8_s32_s32_s32_128x128x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_epi_nosmem");
