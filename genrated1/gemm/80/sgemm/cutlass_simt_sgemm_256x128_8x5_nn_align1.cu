
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


// Gemm operator cutlass_simt_sgemm_256x128_8x5_nn_align1
using cutlass_simt_sgemm_256x128_8x5_nn_align1_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,    // transposed B operand
    float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,    // transposed A operand
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >
,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    5,
    cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// Define named type
struct cutlass_simt_sgemm_256x128_8x5_nn_align1 :
  public cutlass_simt_sgemm_256x128_8x5_nn_align1_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_simt_sgemm_256x128_8x5_nn_align1_operation = new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_simt_sgemm_256x128_8x5_nn_align1>
>("cutlass_simt_sgemm_256x128_8x5_nn_align1");
