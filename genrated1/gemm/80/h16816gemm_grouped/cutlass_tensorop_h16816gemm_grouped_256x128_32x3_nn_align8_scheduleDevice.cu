
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
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Gemm operator cutlass_tensorop_h16816gemm_grouped_256x128_32x3_nn_align8_scheduleDevice
using cutlass_tensorop_h16816gemm_grouped_256x128_32x3_nn_align8_scheduleDevice_base =
  typename cutlass::gemm::kernel::DefaultGemmGrouped<
    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      cutlass::half_t,
      cutlass::half_t
    >
,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// Define named type
struct cutlass_tensorop_h16816gemm_grouped_256x128_32x3_nn_align8_scheduleDevice :
  public cutlass_tensorop_h16816gemm_grouped_256x128_32x3_nn_align8_scheduleDevice_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

auto cutlass_tensorop_h16816gemm_grouped_256x128_32x3_nn_align8_scheduleDevice_operation = new GemmGroupedOperation<
    cutlass::gemm::device::GemmGrouped<cutlass_tensorop_h16816gemm_grouped_256x128_32x3_nn_align8_scheduleDevice>
>("cutlass_tensorop_h16816gemm_grouped_256x128_32x3_nn_align8_scheduleDevice");
