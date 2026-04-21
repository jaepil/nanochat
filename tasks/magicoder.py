"""
ise-uiuc/Magicoder-Evol-Instruct-110K: ~110k evolved code instruction-tuning
examples. Covers multiple languages and diverse coding tasks via WizardCoder-
style instruction evolution.
https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K
"""

from datasets import load_dataset
from tasks.common import Task


class Magicoder(Task):
    """ise-uiuc/Magicoder-Evol-Instruct-110K. ~110k examples, single split."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "Magicoder only has train split"
        self.ds = load_dataset(
            "ise-uiuc/Magicoder-Evol-Instruct-110K", split=split
        ).shuffle(seed=42)
        self.length = len(self.ds)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        instruction = row["instruction"].strip()
        response = row["response"].strip()
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        return {"messages": messages}
