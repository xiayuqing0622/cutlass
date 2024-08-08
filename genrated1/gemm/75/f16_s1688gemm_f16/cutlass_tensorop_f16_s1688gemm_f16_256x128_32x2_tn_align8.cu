
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


// Gemm operator cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tn_align8
using cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tn_align8_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >
,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// Define named type
struct cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tn_align8 :
  public cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tn_align8_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tn_align8_operation = new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tn_align8>
>("cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tn_align8");
