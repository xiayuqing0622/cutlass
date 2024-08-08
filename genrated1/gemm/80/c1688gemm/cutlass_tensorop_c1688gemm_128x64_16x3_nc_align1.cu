
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


// Gemm operator cutlass_tensorop_c1688gemm_128x64_16x3_nc_align1
using cutlass_tensorop_c1688gemm_128x64_16x3_nc_align1_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::complex<float>, cutlass::layout::RowMajor, cutlass::ComplexTransform::kConjugate, 1,    // transposed B operand
    cutlass::complex<float>, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,    // transposed A operand
    cutlass::complex<float>, cutlass::layout::RowMajor,
    cutlass::complex<float>,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    
    cutlass::epilogue::thread::LinearCombination<
      cutlass::complex<float>,
      1,
      cutlass::complex<float>,
      cutlass::complex<float>
    >
,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    cutlass::arch::OpMultiplyAddComplexFastF32
>::GemmKernel;

// Define named type
struct cutlass_tensorop_c1688gemm_128x64_16x3_nc_align1 :
  public cutlass_tensorop_c1688gemm_128x64_16x3_nc_align1_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_c1688gemm_128x64_16x3_nc_align1_operation = new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_c1688gemm_128x64_16x3_nc_align1>
>("cutlass_tensorop_c1688gemm_128x64_16x3_nc_align1");
