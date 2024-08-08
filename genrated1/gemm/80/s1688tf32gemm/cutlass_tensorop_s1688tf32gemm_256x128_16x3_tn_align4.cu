
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


// Gemm operator cutlass_tensorop_s1688tf32gemm_256x128_16x3_tn_align4
using cutlass_tensorop_s1688tf32gemm_256x128_16x3_tn_align4_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 4,    // transposed B operand
    float, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed A operand
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >
,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// Define named type
struct cutlass_tensorop_s1688tf32gemm_256x128_16x3_tn_align4 :
  public cutlass_tensorop_s1688tf32gemm_256x128_16x3_tn_align4_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_s1688tf32gemm_256x128_16x3_tn_align4_operation = new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_s1688tf32gemm_256x128_16x3_tn_align4>
>("cutlass_tensorop_s1688tf32gemm_256x128_16x3_tn_align4");
