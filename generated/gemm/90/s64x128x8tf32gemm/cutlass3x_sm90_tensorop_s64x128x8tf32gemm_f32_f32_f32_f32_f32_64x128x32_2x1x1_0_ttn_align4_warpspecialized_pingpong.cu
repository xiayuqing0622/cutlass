
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



using cutlass3x_sm90_tensorop_s64x128x8tf32gemm_f32_f32_f32_f32_f32_64x128x32_2x1x1_0_ttn_align4_warpspecialized_pingpong_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_64, cute::_128, cute::_32>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    float, cutlass::layout::ColumnMajor, 4,
    float, cutlass::layout::ColumnMajor, 4,
    cutlass::gemm::EpilogueTransposed,
    
    cutlass::epilogue::fusion::LinearCombination<
      float,
      float,
      float,
      float
    >

  >::CollectiveOp;

using cutlass3x_sm90_tensorop_s64x128x8tf32gemm_f32_f32_f32_f32_f32_64x128x32_2x1x1_0_ttn_align4_warpspecialized_pingpong_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    float, cutlass::layout::RowMajor, 4,
    float, cutlass::layout::RowMajor, 4,
    float,
    cute::Shape<cute::_64, cute::_128, cute::_32>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_s64x128x8tf32gemm_f32_f32_f32_f32_f32_64x128x32_2x1x1_0_ttn_align4_warpspecialized_pingpong_epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_s64x128x8tf32gemm_f32_f32_f32_f32_f32_64x128x32_2x1x1_0_ttn_align4_warpspecialized_pingpong
using cutlass3x_sm90_tensorop_s64x128x8tf32gemm_f32_f32_f32_f32_f32_64x128x32_2x1x1_0_ttn_align4_warpspecialized_pingpong_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_s64x128x8tf32gemm_f32_f32_f32_f32_f32_64x128x32_2x1x1_0_ttn_align4_warpspecialized_pingpong_mainloop,
    cutlass3x_sm90_tensorop_s64x128x8tf32gemm_f32_f32_f32_f32_f32_64x128x32_2x1x1_0_ttn_align4_warpspecialized_pingpong_epilogue,
    cutlass::gemm::PersistentScheduler>;

// Define named type
struct cutlass3x_sm90_tensorop_s64x128x8tf32gemm_f32_f32_f32_f32_f32_64x128x32_2x1x1_0_ttn_align4_warpspecialized_pingpong :
  public cutlass3x_sm90_tensorop_s64x128x8tf32gemm_f32_f32_f32_f32_f32_64x128x32_2x1x1_0_ttn_align4_warpspecialized_pingpong_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm90_tensorop_s64x128x8tf32gemm_f32_f32_f32_f32_f32_64x128x32_2x1x1_0_ttn_align4_warpspecialized_pingpong(Manifest &manifest) {



  {
    using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm90_tensorop_s64x128x8tf32gemm_f32_f32_f32_f32_f32_64x128x32_2x1x1_0_ttn_align4_warpspecialized_pingpong>;
    manifest.append(
      new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_s64x128x8tf32gemm_f32_f32_f32_f32_f32_64x128x32_2x1x1_0_ttn_align4_warpspecialized_pingpong"));
  }



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

