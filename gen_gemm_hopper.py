
import os
import re

def SubstituteTemplate(template, values):
  text = template
  changed = True
  while changed:
    changed = False
    for key, value in values.items():
      regex = "\\$\\{%s\\}" % key
      newtext = re.sub(regex, value, text)
      if newtext != text:
        changed = True
      text = newtext
  return text

class EmitGemmUniversal3xInstance:
    ''' Responsible for emitting a CUTLASS 3.x template definition'''
    def __init__(self):
        self.gemm_hopper_template = """

/*! \\file
    \\brief Simple Hopper GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture

    Examples:

      $ ./gemm_hopper --m=2048 --n=2048 --k=2048 
*/


#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = ${element_a};                                   // Element type for A matrix operand
using         LayoutA     = ${layout_a};                                    // Layout type for A matrix operand
constexpr int AlignmentA  = ${align_a};//128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = ${element_b};                                   // Element type for B matrix operand
using         LayoutB     = ${layout_b};                                    // Layout type for B matrix operand
constexpr int AlignmentB  = ${align_b};    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
using         ElementC    = ${element_c};                            // Element type for C matrix operands
using         LayoutC     = ${layout_c};                       // Layout type for C matrix operands
constexpr int AlignmentC  = ${align_c};    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ${element_d};                                      // Element type for D matrix operands
using         LayoutD     = ${layout_d};                                        // Layout type  D matrix operands
constexpr int AlignmentD  = ${align_d};                                     // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)


// Core kernel configurations
using ElementAccumulator  = ${element_accumulator};                                          // Element type for internal accumulation
using ElementEpilogue = ${element_epilogue};                                          // Element type for internal computation
using ArchTag             = ${arch};                                // Tag indicating the minimum SM that supports the intended feature
using OperatorClassEpi       = ${opcode_class_epi};                 // Operator epilogue class tag
using OperatorClassMain       = ${opcode_class_main};                 // Operator mainloop class tag
using TileShape           = ${tile_shape_epi};                           // Threadblock-level tile size
using TileShapeMain       = ${tile_shape_main};                           // Threadblock-level tile size
using ClusterShape        = ${cluster_shape};                                // Shape of the threadblocks in a cluster
using EpilogueTileType    = ${epi_tile_mn}; //default
using KernelScheduleType =  ${kernel_schedule};
using EpilogueScheduleType = ${epilogue_schedule};
using FusionOpOrCallbacks = ${epilogue_functor}<ElementD,ElementEpilogue,ElementC,ElementEpilogue>;
using TileSchedulerType = ${tile_scheduler};

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, OperatorClassEpi,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementEpilogue,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueScheduleType,
    FusionOpOrCallbacks

  >::CollectiveOp;

using StageCountType = ${stages};

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClassMain,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShapeMain, ClusterShape,
    StageCountType,
    KernelScheduleType
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int, int>, // Indicates ProblemShape <m,n,k,l>
    CollectiveMainloop,
    CollectiveEpilogue,
    TileSchedulerType
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

// Extract information from Gemm kernel.
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_C;
std::vector<int64_t> offset_D;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

cutlass::DeviceAllocation<const typename Gemm::ElementA *> ptr_A;
cutlass::DeviceAllocation<const typename Gemm::ElementB *> ptr_B;
cutlass::DeviceAllocation<const typename Gemm::ElementC *> ptr_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_ref_D;


#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k,l ;

  Options():
    help(false),
    m(5120), n(4096), k(4096), l(1),
    alpha(1.f), beta(0.f),
    iterations(1000)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "48_hopper_warp_specialized_gemm\\n\\n"
      << "  Hopper FP32 GEMM using a Warp Specialized kernel.\\n\\n"
      << "Options:\\n\\n"
      << "  --help                      If specified, displays this usage statement\\n\\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\\n"
      << "  --l=<int>                   Sets the L extent of the GEMM\\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\\n"
      << "  --beta=<f32>                Epilogue scalar beta\\n\\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\\n\\n";

    out
      << "\\n\\nExamples:\\n\\n"
      << "$ " << "48_hopper_warp_specialized_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \\n\\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k * l;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

/// Allocates device-side data
void allocate(const Options &options) {
  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;

  for (int32_t i = 0; i < options.l; ++i) {

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);

    int64_t elements_A = options.m * options.k;
    int64_t elements_B = options.k * options.n;
    int64_t elements_C = options.m * options.n;
    int64_t elements_D = options.m * options.n;

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_ref_D.reset(total_elements_D);
}
/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

  //
  // Assign pointers
  //

  std::vector<ElementA *> ptr_A_host(options.l);
  std::vector<ElementB *> ptr_B_host(options.l);
  std::vector<ElementC *> ptr_C_host(options.l);
  std::vector<ElementC *> ptr_D_host(options.l);

  for (int32_t i = 0; i < options.l; ++i) {
    ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
    ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
  }

  ptr_A.reset(options.l);
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(options.l);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(options.l);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.l);
  ptr_D.copy_from_host(ptr_D_host.data());

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

  typename Gemm::Arguments arguments{
    options.l>1 ? cutlass::gemm::GemmUniversalMode::kBatched : cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta}, //epilogue.thread 
    block_C.get(), stride_C, block_D.get(), stride_D}
  };

  //arguments.scheduler.raster_order =  ${raster_order}; // choose from[RasterOrderOptions::AlongN, RasterOrderOptions::AlongM, RasterOrderOptions::Heuristic]   
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8) 
  //arguments.scheduler.max_swizzle_size = ${max_swizzle_size};

  return arguments;
}

bool verify(const Options &options) {
  bool passed = true;
  for (int32_t i = 0; i < options.l; ++i) {
    cutlass::TensorRef ref_A(block_A.get() + offset_A.at(i), Gemm::LayoutA::packed({options.m, options.k}));
    cutlass::TensorRef ref_B(block_B.get() + offset_B.at(i), Gemm::LayoutB::packed({options.k, options.n}));
    cutlass::TensorRef ref_C(block_C.get() + offset_C.at(i), Gemm::LayoutC::packed({options.m, options.n}));
    cutlass::TensorRef ref_D(block_ref_D.get() + offset_D.at(i), Gemm::LayoutD::packed({options.m, options.n}));

    //
    // Compute reference output
    //

    // Create instantiation for device reference gemm kernel
    DeviceGemmReference gemm_reference;

    // Launch device reference gemm kernel
    gemm_reference(
      {options.m, options.n, options.k},
      ElementAccumulator(options.alpha),
      ref_A,
      ref_B,
      ElementAccumulator(options.beta),
      ref_C,
      ref_D);

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    passed &= cutlass::reference::device::BlockCompareEqual(block_ref_D.get() + offset_D.at(i), block_D.get() + offset_D.at(i), options.m * options.n);
  }
  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  allocate(options);
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size : " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    std::cout << "  Avg runtime : " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS : " << result.gflops << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major < 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\\n";
    return 0;
  }
  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  run<Gemm>(options);
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
"""
    def emit(self, file_path, output_dir):
        with open(file_path, 'r') as file:
            lines = file.readlines()

            element_d = lines[36].strip().split(",")[0]
            element_epilogue = lines[37].strip().split(",")[0]
            element_c = lines[38].strip().split(",")[0]
            line_26=lines[26].strip().split(",")
            arch, opcode_class_epi = line_26[0], line_26[1]
            tile_shape_epi = lines[27].strip()[:-1]
            cluster_shape = lines[28].strip()[:-1]
            epi_tile_mn = lines[29].strip().split(",")[0] #epi tile type
            element_accumulator = lines[30].strip().split(",")[0]
            line_31=lines[31].strip().split(",")
            layout_c, align_c = line_31[1], line_31[2]
            line_32=lines[32].strip().split(",")
            layout_d, align_d = line_32[1], line_32[2]
            epilogue_schedule = lines[33].strip().split(",")[0]
            epilogue_functor = lines[35].strip()[:-1]
            opcode_class_main = lines[46].strip().split(",")[1]
            line_47=lines[47].strip().split(",")
            element_a, layout_a, align_a = line_47[0], line_47[1], line_47[2]
            line_48=lines[48].strip().split(",")
            element_b, layout_b, align_b = line_48[0], line_48[1], line_48[2]
            tile_shape_main = lines[50].strip()[:-1]
            stages = lines[52].strip().split(",")[0]
            if "typename" in stages:
                start_index = stages.index("typename") + len("typename ")
                end_index = stages.index("::SharedStorage")
                substring_to_replace = stages[start_index:end_index]
                stages = stages.replace(substring_to_replace, "CollectiveEpilogue")
            kernel_schedule = lines[53].strip()
            tile_scheduler = lines[61].strip()[:-2]
            name = lines[83].strip().split("\"")[1] + ".cu"
        values = {
            "element_d": element_d,
            "element_epilogue": element_epilogue,
            "element_c": element_c,
            "arch": arch,
            "opcode_class_epi": opcode_class_epi,
            "tile_shape_epi": tile_shape_epi,
            "cluster_shape": cluster_shape,
            "epi_tile_mn": epi_tile_mn,
            "element_accumulator": element_accumulator,
            "layout_c": layout_c,
            "align_c": align_c,
            "layout_d": layout_d,
            "align_d": align_d,
            "epilogue_schedule": epilogue_schedule,
            "epilogue_functor": epilogue_functor,
            "opcode_class_main": opcode_class_main,
            "element_a": element_a,
            "layout_a": layout_a,
            "align_a": align_a,
            "element_b": element_b,
            "layout_b": layout_b,
            "align_b": align_b,
            "tile_shape_main": tile_shape_main,
            "stages": stages,
            "kernel_schedule": kernel_schedule,
            "tile_scheduler": tile_scheduler
        }

        code = SubstituteTemplate(self.gemm_hopper_template, values)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, name), 'w') as file:
            file.write(code)



def list_files_with_prefix(directory, prefix):
    # List all files in the given directory
    files = os.listdir(directory)
    
    # Filter files that start with the specified prefix
    matching_files = [file for file in files if file.startswith(prefix)]
    
    return matching_files




gemm_90_directory_path = "./build/tools/library/generated/gemm/90/"
sub_directories = os.listdir(gemm_90_directory_path)

for sub_directory in sub_directories:
    directory_path = os.path.join(gemm_90_directory_path, sub_directory)
    if not os.path.isdir(directory_path):
        continue
    prefix = "cutlass3x"
    matching_files = list_files_with_prefix(directory_path, prefix)
    output_dir = os.path.join("./examples/48_hopper_warp_specialized_gemm/90", sub_directory)
    for file in matching_files:
        emit_gemm3x = EmitGemmUniversal3xInstance()
        emit_gemm3x.emit(os.path.join(directory_path, file), output_dir)
# directory_path = os.path.join(gemm_90_directory_path,)
# prefix = "cutlass3x"
# matching_files = list_files_with_prefix(directory_path, prefix)
# for file in matching_files:
#     emit_gemm3x = EmitGemmUniversal3xInstance()
#     emit_gemm3x.emit(os.path.join(directory_path, file), "./examples/48_hopper_warp_specialized_gemm/bf16_s64x128x16gemm_bf16")
  














