import torch
# gt_embeddings_dict = torch.load('/data/espnet/egs2/libritts/tts1/dump/raw/train-clean-100_phn/semantics_ave.pt') # 加载文本嵌入的字典
gt_embeddings_dict = torch.load('/data/vitsGPT/vits/filelists/ljs_audio_gt_mat_text_t5120.pt') # 加载文本嵌入的字典
print(gt_embeddings_dict)
# print(len(gt_embeddings_dict))


# def count_rows_in_csv(file_path):
#     with open(file_path, 'r') as file:
#         return sum(1 for _ in file)
# file_path = '/data/vitsGPT/datasets/LJSpeech-1.1/metadata.csv'
# number_of_rows = count_rows_in_csv(file_path)
# print(f"The CSV file has {number_of_rows} rows.")


import csv
# 准备写入 CSV 文件
# with open('/data/espnet/egs2/libritts/tts1/dump/raw/train-clean-100_phn/output_check.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     # 对于字典中的每个项
#     for key, tensor in gt_embeddings_dict.items():
#         # 获取 tensor 的头尾各 5 个值
#         values = list(tensor[:5].numpy()) + list(tensor[-5:].numpy())
#         # 写入 CSV 文件
#         writer.writerow([key] + values)
        
