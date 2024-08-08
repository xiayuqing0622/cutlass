
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


// Gemm operator cutlass_sm90_tensorop_z1684gemm_cf64_cf64_cf64_cf64_cf64_128x64x8_1x1x1_3_cnn_align1
using cutlass_sm90_tensorop_z1684gemm_cf64_cf64_cf64_cf64_cf64_128x64x8_1x1x1_3_cnn_align1_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::complex<double>, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,    // transposed B operand
    cutlass::complex<double>, cutlass::layout::RowMajor, cutlass::ComplexTransform::kConjugate, 1,    // transposed A operand
    cutlass::complex<double>, cutlass::layout::RowMajor,
    cutlass::complex<double>,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<128, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<16, 8, 4>,
    
    cutlass::epilogue::thread::LinearCombination<
      cutlass::complex<double>,
      1,
      cutlass::complex<double>,
      cutlass::complex<double>
    >
,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    cutlass::arch::OpMultiplyAddComplex
>::GemmKernel;

// Define named type
struct cutlass_sm90_tensorop_z1684gemm_cf64_cf64_cf64_cf64_cf64_128x64x8_1x1x1_3_cnn_align1 :
  public cutlass_sm90_tensorop_z1684gemm_cf64_cf64_cf64_cf64_cf64_128x64x8_1x1x1_3_cnn_align1_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_sm90_tensorop_z1684gemm_cf64_cf64_cf64_cf64_cf64_128x64x8_1x1x1_3_cnn_align1_operation = new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_sm90_tensorop_z1684gemm_cf64_cf64_cf64_cf64_cf64_128x64x8_1x1x1_3_cnn_align1>
>("cutlass_sm90_tensorop_z1684gemm_cf64_cf64_cf64_cf64_cf64_128x64x8_1x1x1_3_cnn_align1");
