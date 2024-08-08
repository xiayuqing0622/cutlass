
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


// Gemm operator cutlass_tensorop_i168256andgemm_b1_256x128_512x3_tn_align128
using cutlass_tensorop_i168256andgemm_b1_256x128_512x3_tn_align128_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 128,    // transposed B operand
    cutlass::uint1b_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 128,    // transposed A operand
    int32_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 512>,
    cutlass::gemm::GemmShape<64, 64, 512>,
    cutlass::gemm::GemmShape<16, 8, 256>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >
,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    cutlass::arch::OpAndPopc
>::GemmKernel;

// Define named type
struct cutlass_tensorop_i168256andgemm_b1_256x128_512x3_tn_align128 :
  public cutlass_tensorop_i168256andgemm_b1_256x128_512x3_tn_align128_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_i168256andgemm_b1_256x128_512x3_tn_align128_operation = new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_i168256andgemm_b1_256x128_512x3_tn_align128>
>("cutlass_tensorop_i168256andgemm_b1_256x128_512x3_tn_align128");
