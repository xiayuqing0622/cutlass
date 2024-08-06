file_path = "/home/aiscuser/yuqxia/cutlass/build/tools/library/generated/gemm/90/bf16_s64x128x16gemm_bf16/cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x128x64_2x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tma.cu"
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
    tile_shape_main = lines[49].strip()[:-1]
    stages = lines[52].strip().split(",")[0]
    if "typename" in stages:
        start_index = stages.index("typename") + len("typename ")
        end_index = stages.index("::SharedStorage")
        substring_to_replace = stages[start_index:end_index]
        stages = stages.replace(substring_to_replace, "CollectiveEpilogue")
    kernel_schedule = lines[53].strip()
    tile_scheduler = lines[61].strip()[:-2]

    name = lines[83].strip().split("\"")[1]

print("element_d: ", element_d)
print("element_epilogue: ", element_epilogue)
print("element_c: ", element_c)
print("arch: ", arch)
print("opcode_class_epi: ", opcode_class_epi)
print("tile_shape_epi: ", tile_shape_epi)
print("cluster_shape: ", cluster_shape)
print("epi_tile_mn: ", epi_tile_mn)
print("element_accumulator: ", element_accumulator)
print("layout_c: ", layout_c)
print("align_c: ", align_c)
print("layout_d: ", layout_d)
print("align_d: ", align_d)
print("epilogue_schedule: ", epilogue_schedule)
print("epilogue_functor: ", epilogue_functor)
print("opcode_class_main: ", opcode_class_main)
print("element_a: ", element_a)
print("layout_a: ", layout_a)
print("align_a: ", align_a)
print("element_b: ", element_b)
print("layout_b: ", layout_b)
print("align_b: ", align_b)
print("tile_shape_main: ", tile_shape_main)
print("stages: ", stages)
print("kernel_schedule: ", kernel_schedule)
print("tile_scheduler: ", tile_scheduler)
print("name: ", name)