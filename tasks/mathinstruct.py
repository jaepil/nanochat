"""
TIGER-Lab/MathInstruct: ~260k math instruction-tuning examples covering
arithmetic, algebra, geometry, probability, and word problems. Diverse
sources (GSM8K, MATH, OpenMathInstruct, Aqua-RAT, etc.) formatted as
instruction/output pairs.
https://huggingface.co/datasets/TIGER-Lab/MathInstruct
"""

from datasets import load_dataset
from tasks.common import Task


class MathInstruct(Task):
    """TIGER-Lab/MathInstruct. ~260k examples, single train split."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "MathInstruct only has train split"
        self.ds = load_dataset("TIGER-Lab/MathInstruct", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        instruction = row["instruction"].strip()
        output = row["output"].strip()
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
        return {"messages": messages}
