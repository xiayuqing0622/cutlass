
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


// Gemm operator cutlass_tensorop_u8_i8816gemm_u8_256x128_64x2_tn_align16
using cutlass_tensorop_u8_i8816gemm_u8_256x128_64x2_tn_align16_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    uint8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,    // transposed B operand
    uint8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,    // transposed A operand
    uint8_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombinationClamp<
      uint8_t,
      16,
      int32_t,
      float
    >
,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    cutlass::arch::OpMultiplyAddSaturate
>::GemmKernel;

// Define named type
struct cutlass_tensorop_u8_i8816gemm_u8_256x128_64x2_tn_align16 :
  public cutlass_tensorop_u8_i8816gemm_u8_256x128_64x2_tn_align16_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_u8_i8816gemm_u8_256x128_64x2_tn_align16_operation = new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_u8_i8816gemm_u8_256x128_64x2_tn_align16>
>("cutlass_tensorop_u8_i8816gemm_u8_256x128_64x2_tn_align16");
