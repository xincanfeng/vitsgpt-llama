# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import fire
from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs = [
        [{"role": "system", "content": "Always answer with a single word"},
         {"role": "user", "content": "what is the fine-grained emotion of above sentence?"},
         {"role": "user", "content": "what is the aspect of that emotion?"},],
        [{"role": "user", "content": "what is the emotion of the following sentence: I have an apple.?"}],
        [{"role": "user", "content": "what is the fine-grained emotion of above sentence?"},
         {"role": "user", "content": "What is the aspect of that emotion?"},],
        [{"role": "system", "content": "You are an expert who can understand any emotion, no matter simple or complex emotion, in the text. Based on the text, you will delve into the emotions conveyed and the aspect of that emotion by employing any means you can, including inferring its background, and summarize it for me in a single sentence. You can convey the emotion and aspect of the text very well in a expressive sentence.",},
         {"role": "user", "content": "what is the fine-grained emotion of above sentence?"},
         {"role": "user", "content": "what is the aspect of that emotion?"},],
    ]

    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
