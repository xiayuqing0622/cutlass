import os

def delete_files_starting_with_all(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("all"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

# Replace '/path/to/your/directory' with the path to the directory you want to clean up
# root_directory = './generated'
# delete_files_starting_with_all(root_directory)



def get_filename(file_path):
    return os.path.basename(file_path).split('.')[0]

def process_gemm_file(file_path, filename):
    filename = get_filename(filename)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for i in range(len(lines)):
        if lines[i].startswith("namespace cutlass {"):
            if lines[i+9].strip() == "{":
                assert lines[i+11].strip().startswith("manifest.append(")
                lines[i+11] = lines[i+11].replace("manifest.append(", f"auto {filename}_operation = ").strip() + "\n"
                assert lines[i+12].strip().endswith(");")
                lines[i+12] = lines[i+12].replace(");", ";").strip() + "\n"
                new_lines = lines[:i] + [lines[i+11], lines[i+12]]
                break
            else:
                assert lines[i+9].strip().startswith("manifest.append(")
                lines[i+9] = lines[i+9].replace("manifest.append(", f"auto {filename}_operation = ").strip() + "\n"
                if lines[i+9].strip().endswith(");"):
                    lines[i+9] = lines[i+9].replace(");", ";").strip() + "\n"
                    new_lines = lines[:i] + [lines[i+9]]
                    break
                elif lines[i+10].strip().endswith(");"):
                    lines[i+10] = lines[i+10].replace(");", ";").strip() + "\n"
                    new_lines = lines[:i] + [lines[i+9], lines[i+10]]
                    break
                elif lines[i+11].strip().endswith(");"):
                    lines[i+11] = lines[i+11].replace(");", ";").strip() + "\n"
                    new_lines = lines[:i] + [lines[i+9], lines[i+10], lines[i+11]]
                    break
    
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

def process_gemm_directory(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                process_gemm_file(file_path, filename)
                  

def process_conv2d_file(file_path, filename):
    filename = get_filename(filename)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for i in range(len(lines)):
        if lines[i].startswith("namespace cutlass {"):
            assert lines[i+9].strip().startswith("manifest.append(")
            lines[i+9] = lines[i+9].replace("manifest.append(", f"auto {filename}_operation = ").strip() + "\n"
            assert lines[i+13].strip().endswith(");")
            lines[i+13] = lines[i+13].replace(");", ";").strip() + "\n"
            new_lines = lines[:i] + lines[i+6:i+14]
            break
    
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

def process_conv2d_directory(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                process_conv2d_file(file_path, filename)


def process_rank2k_file(file_path, filename):
    filename = get_filename(filename)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for i in range(len(lines)):
        if lines[i].startswith("namespace cutlass {"):
            assert lines[i+9].strip().startswith("manifest.append(")
            lines[i+9] = lines[i+9].replace("manifest.append(", f"auto {filename}_operation = ").strip() + "\n"
            assert lines[i+11].strip().endswith(");")
            lines[i+11] = lines[i+11].replace(");", ";").strip() + "\n"
            new_lines = lines[:i] + lines[i+9:i+12]
            break
    
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

def process_rank2k_directory(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                process_rank2k_file(file_path, filename)
# Replace '/path/to/your/directory' with the path to the directory you want to process

# process_gemm_directory("./genrated1/gemm")
# process_conv2d_directory("./genrated1/conv2d")
# process_conv2d_directory("./genrated1/conv3d")     
# process_rank2k_directory("./genrated1/rank_2k")   
# process_rank2k_directory("./genrated1/rank_k")     
# process_rank2k_directory("./genrated1/symm")  
# process_rank2k_directory("./genrated1/trmm")


import os

def count_files(root_dir):
    file_count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        file_count += len(filenames)
    return file_count

# Replace '/path/to/your/directory' with the path to the directory you want to count files in
root_directory = './genrated1'
total_files = count_files(root_directory)
print(f"Total number of files: {total_files}")
