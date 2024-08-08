
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


// Gemm operator cutlass_tensorop_bf16_s16816gemm_u8_bf16_128x128_64x4_tn_align16
using cutlass_tensorop_bf16_s16816gemm_u8_bf16_128x128_64x4_tn_align16_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::bfloat16_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    uint8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,    // transposed A operand
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      cutlass::bfloat16_t,
      8,
      float,
      float
    >
,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    cutlass::arch::OpMultiplyAddMixedInputUpcast
>::GemmKernel;

// Define named type
struct cutlass_tensorop_bf16_s16816gemm_u8_bf16_128x128_64x4_tn_align16 :
  public cutlass_tensorop_bf16_s16816gemm_u8_bf16_128x128_64x4_tn_align16_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_bf16_s16816gemm_u8_bf16_128x128_64x4_tn_align16_operation = new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_bf16_s16816gemm_u8_bf16_128x128_64x4_tn_align16>
>("cutlass_tensorop_bf16_s16816gemm_u8_bf16_128x128_64x4_tn_align16");
