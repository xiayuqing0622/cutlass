
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
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Gemm operator cutlass_tensorop_s4_i16864gemm_s4_256x128_128x3_tn_align32
using cutlass_tensorop_s4_i16864gemm_s4_256x128_128x3_tn_align32_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 32,    // transposed B operand
    cutlass::int4b_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 32,    // transposed A operand
    cutlass::int4b_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    
    cutlass::epilogue::thread::LinearCombinationClamp<
      cutlass::int4b_t,
      16,
      int32_t,
      float
    >
,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    cutlass::arch::OpMultiplyAddSaturate
>::GemmKernel;

// Define named type
struct cutlass_tensorop_s4_i16864gemm_s4_256x128_128x3_tn_align32 :
  public cutlass_tensorop_s4_i16864gemm_s4_256x128_128x3_tn_align32_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_s4_i16864gemm_s4_256x128_128x3_tn_align32(Manifest &manifest) {



  manifest.append(new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_s4_i16864gemm_s4_256x128_128x3_tn_align32>
    >("cutlass_tensorop_s4_i16864gemm_s4_256x128_128x3_tn_align32"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

