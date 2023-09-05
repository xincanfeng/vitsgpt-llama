# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import fire
import torch
import csv
from llama import Llama


output_file_name = 'ljs_audio_gt_last'

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    input_file: str = '/data/vitsGPT/datasets/LJSpeech-1.1/metadata_copy10.csv',
    output_file: str = f"/data/vitsGPT/vits/filelists/{output_file_name}_5120_test.pt",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_gen_len: int = 64,
    max_batch_size: int = 8,
):
    
    rank = os.environ.get('RANK', 0) # 获取当前进程的 rank

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    with open(input_file, 'r') as f:
        reader = csv.reader(f, delimiter='|')
        data = [(row[0], row[1]) for row in reader]  # 保存第一列和第二列的内容

    temp_files = []
    # 按批次处理文本
    for idx, batch in enumerate(chunks(data, max_batch_size)):
        keys = ["DUMMY1/" + item[0] + ".wav" for item in batch]  # 修改此行以适配正确的音频链接
        prompts = [item[1] for item in batch]

        results = generator.text_completion(
            prompts,
            max_gen_len=max_seq_len,
            temperature=temperature,
            top_p=top_p,
        )

        # 紧接着generation，调用 get_promt_last_token_embedding 方法
        # h_last_real_token_b, h_ave_real_token_b, h_last_real_token_slt, h_ave_real_token_slt, h_pca_real_token_slt = generator.get_text_prompt_token_embedding()
        _, _, h_last_real_token_slt, _, _ = generator.get_text_prompt_token_embedding()
        # _, _, _, h_ave_real_token_slt, _ = generator.get_text_prompt_token_embedding()
        # _, _, _, _, h_pca_real_token_slt = generator.get_text_prompt_token_embedding()

        gt_embeddings = h_last_real_token_slt
        # gt_embeddings = h_ave_real_token_slt
        # gt_embeddings = h_pca_real_token_slt

        batch_output_dict = {key: embedding.cpu() for key, embedding in zip(keys, gt_embeddings)}
        
        # 为每个进程使用不同的临时文件名
        temp_filename = f"{output_file_name}_temp_{rank}_{idx}.pt"
        torch.save(batch_output_dict, temp_filename)
        temp_files.append(temp_filename)
            
        for key, prompt, result, embedding in zip(keys, prompts, results, gt_embeddings):
            print(f"geting embedding for {output_file_name}:")
            print(key)
            print(prompt)
            print(result)
            print(embedding[:10])
            print("\n==================================\n")

    # Combine all the batch outputs into a final output
    final_output_dict = {}
    for temp_file in temp_files:
        batch_data = torch.load(temp_file)
        final_output_dict.update(batch_data)
        # os.remove(temp_file)  # Delete the temporary file

    torch.save(final_output_dict, output_file)

if __name__ == "__main__":
    fire.Fire(main)