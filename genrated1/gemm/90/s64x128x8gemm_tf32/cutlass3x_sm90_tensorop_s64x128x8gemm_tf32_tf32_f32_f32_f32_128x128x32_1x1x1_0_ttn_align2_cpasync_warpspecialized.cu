
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



using cutlass3x_sm90_tensorop_s64x128x8gemm_tf32_tf32_f32_f32_f32_128x128x32_1x1x1_0_ttn_align2_cpasync_warpspecialized_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_32>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    float, cutlass::layout::ColumnMajor, 1,
    float, cutlass::layout::ColumnMajor, 1,
    cutlass::gemm::EpilogueTransposed,
    
    cutlass::epilogue::fusion::LinearCombination<
      float,
      float,
      float,
      float
    >

  >::CollectiveOp;

using cutlass3x_sm90_tensorop_s64x128x8gemm_tf32_tf32_f32_f32_f32_128x128x32_1x1x1_0_ttn_align2_cpasync_warpspecialized_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::tfloat32_t, cutlass::layout::RowMajor, 2,
    cutlass::tfloat32_t, cutlass::layout::RowMajor, 2,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_32>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_s64x128x8gemm_tf32_tf32_f32_f32_f32_128x128x32_1x1x1_0_ttn_align2_cpasync_warpspecialized_epilogue::SharedStorage))>,
    cutlass::gemm::KernelCpAsyncWarpSpecialized
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_s64x128x8gemm_tf32_tf32_f32_f32_f32_128x128x32_1x1x1_0_ttn_align2_cpasync_warpspecialized
using cutlass3x_sm90_tensorop_s64x128x8gemm_tf32_tf32_f32_f32_f32_128x128x32_1x1x1_0_ttn_align2_cpasync_warpspecialized_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_s64x128x8gemm_tf32_tf32_f32_f32_f32_128x128x32_1x1x1_0_ttn_align2_cpasync_warpspecialized_mainloop,
    cutlass3x_sm90_tensorop_s64x128x8gemm_tf32_tf32_f32_f32_f32_128x128x32_1x1x1_0_ttn_align2_cpasync_warpspecialized_epilogue,
    cutlass::gemm::PersistentScheduler>;

// Define named type
struct cutlass3x_sm90_tensorop_s64x128x8gemm_tf32_tf32_f32_f32_f32_128x128x32_1x1x1_0_ttn_align2_cpasync_warpspecialized :
  public cutlass3x_sm90_tensorop_s64x128x8gemm_tf32_tf32_f32_f32_f32_128x128x32_1x1x1_0_ttn_align2_cpasync_warpspecialized_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass3x_sm90_tensorop_s64x128x8gemm_tf32_tf32_f32_f32_f32_128x128x32_1x1x1_0_ttn_align2_cpasync_warpspecialized_operation =
new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_s64x128x8gemm_tf32_tf32_f32_f32_f32_128x128x32_1x1x1_0_ttn_align2_cpasync_warpspecialized");
