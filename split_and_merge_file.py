# def split_file_into_n(input_file_path, n):
#     with open(input_file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     total_lines = len(lines)
#     part_size = total_lines // n

#     for i in range(n):
#         start_index = i * part_size
#         end_index = (i + 1) * part_size if i < n - 1 else None  # 让最后一个文件包含所有剩余的行

#         with open(f"{input_file_path}_split_{i+1}", 'w', encoding='utf-8') as f:
#             f.writelines(lines[start_index:end_index])

# # 使用方法
# split_file_into_n("/data/espnet/egs2/libritts/tts1/data/train-clean-100/text", 4)


import torch

def merge_pt_files(input_file_paths, output_file_path):
    merged_state_dict = {}
    
    for file_path in input_file_paths:
        state_dict = torch.load(file_path)
        merged_state_dict.update(state_dict)
    
    torch.save(merged_state_dict, output_file_path)

# 使用方法
output_file = "/data/espnet/egs2/libritts/tts1/dump/raw/train-clean-100_phn/semantics_pca"
merge_pt_files(
    [f"{output_file}_temp_1.pt", 
    f"{output_file}_temp_2.pt", 
    f"{output_file}_temp_3.pt",
    f"{output_file}_temp_4.pt"],
    f"{output_file}.pt")
