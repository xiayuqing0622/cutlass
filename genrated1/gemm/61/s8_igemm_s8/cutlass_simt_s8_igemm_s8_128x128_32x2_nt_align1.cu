
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


// Gemm operator cutlass_simt_s8_igemm_s8_128x128_32x2_nt_align1
using cutlass_simt_s8_igemm_s8_128x128_32x2_nt_align1_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,    // transposed B operand
    int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,    // transposed A operand
    int8_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm61,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    
    cutlass::epilogue::thread::LinearCombinationClamp<
      int8_t,
      1,
      int32_t,
      int32_t
    >
,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// Define named type
struct cutlass_simt_s8_igemm_s8_128x128_32x2_nt_align1 :
  public cutlass_simt_s8_igemm_s8_128x128_32x2_nt_align1_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_simt_s8_igemm_s8_128x128_32x2_nt_align1_operation = new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_simt_s8_igemm_s8_128x128_32x2_nt_align1>
>("cutlass_simt_s8_igemm_s8_128x128_32x2_nt_align1");
