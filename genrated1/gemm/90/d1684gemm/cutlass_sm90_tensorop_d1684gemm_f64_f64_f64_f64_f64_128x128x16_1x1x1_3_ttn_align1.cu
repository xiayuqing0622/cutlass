
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


// Gemm operator cutlass_sm90_tensorop_d1684gemm_f64_f64_f64_f64_f64_128x128x16_1x1x1_3_ttn_align1
using cutlass_sm90_tensorop_d1684gemm_f64_f64_f64_f64_f64_128x128x16_1x1x1_3_ttn_align1_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    double, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,    // transposed B operand
    double, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,    // transposed A operand
    double, cutlass::layout::RowMajor,
    double,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 4>,
    
    cutlass::epilogue::thread::LinearCombination<
      double,
      1,
      double,
      double
    >
,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// Define named type
struct cutlass_sm90_tensorop_d1684gemm_f64_f64_f64_f64_f64_128x128x16_1x1x1_3_ttn_align1 :
  public cutlass_sm90_tensorop_d1684gemm_f64_f64_f64_f64_f64_128x128x16_1x1x1_3_ttn_align1_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_sm90_tensorop_d1684gemm_f64_f64_f64_f64_f64_128x128x16_1x1x1_3_ttn_align1_operation = new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_sm90_tensorop_d1684gemm_f64_f64_f64_f64_f64_128x128x16_1x1x1_3_ttn_align1>
>("cutlass_sm90_tensorop_d1684gemm_f64_f64_f64_f64_f64_128x128x16_1x1x1_3_ttn_align1");
