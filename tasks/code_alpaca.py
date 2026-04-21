"""
CodeAlpaca-20k dataset for code-focused SFT.
https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k

Fields: instruction, input, output.
"""

from datasets import load_dataset
from tasks.common import Task


class CodeAlpaca(Task):
    """sahil2801/CodeAlpaca-20k. ~20k rows, single split."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "CodeAlpaca only has train split"
        self.ds = load_dataset("sahil2801/CodeAlpaca-20k", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        instruction = row["instruction"].strip()
        input_text = row.get("input", "").strip()
        output = row["output"].strip()

        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        return {"messages": messages}
