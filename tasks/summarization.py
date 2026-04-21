"""Summarization task classes: XSum, CNN/DailyMail, SAMSum, XL-Sum (EN).

All produce single-turn conversations: user=document, assistant=summary.
Each dataset has different column names, so we parametrize.
"""

from datasets import load_dataset
from tasks.common import Task


def _truncate_words(text, max_words):
    """Truncate text to at most max_words (whitespace split)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _make_conv(document, summary, instruction_suffix=None, max_doc_words=800):
    """Build the {messages: [...]} conversation.
    Documents are truncated from the front to at most ``max_doc_words`` words
    to keep the assistant-token ratio well above the gradient-variance floor.
    The instruction is placed at the END of the user turn so the final user
    token carries a strong "produce summary next" cue — this eliminates the
    mode-collapse to "<|assistant_end|>" that prefix-instruction format showed.
    """
    doc = _truncate_words(document.strip(), max_doc_words)
    user_content = doc
    if instruction_suffix:
        user_content = f"{doc}\n\n{instruction_suffix}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": summary.strip()},
        ]
    }


class XSum(Task):
    """EdinburghNLP/xsum: 204k BBC article -> 1-sentence summary. MIT."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "validation", "test"), f"XSum split: {split}"
        self.ds = load_dataset(
            "EdinburghNLP/xsum", split=split, trust_remote_code=True
        ).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        return _make_conv(
            row["document"],
            row["summary"],
            instruction_suffix="Summary (one sentence):",
        )


class CNNDailyMail(Task):
    """abisee/cnn_dailymail v3.0.0: 287k article -> multi-sentence highlights. Apache 2.0."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "validation", "test"), f"CNN/DM split: {split}"
        self.ds = load_dataset(
            "abisee/cnn_dailymail", "3.0.0", split=split, trust_remote_code=True
        ).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        return _make_conv(
            row["article"],
            row["highlights"],
            instruction_suffix="Multi-sentence summary:",
        )


class SAMSum(Task):
    """Samsung/samsum: 16k dialogue -> summary. CC-BY-NC 4.0 (non-commercial)."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "validation", "test"), f"SAMSum split: {split}"
        self.ds = load_dataset(
            "Samsung/samsum", split=split, trust_remote_code=True
        ).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        return _make_conv(
            row["dialogue"],
            row["summary"],
            instruction_suffix="Dialogue summary:",
        )


class XLSumEN(Task):
    """csebuetnlp/xlsum English subset: ~300k BBC multilingual summaries, EN only.
    CC-BY-NC-SA 4.0 (non-commercial).
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "validation", "test"), f"XLSum split: {split}"
        self.ds = load_dataset(
            "csebuetnlp/xlsum", "english", split=split, trust_remote_code=True
        ).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        return _make_conv(
            row["text"],
            row["summary"],
            instruction_suffix="Article summary:",
        )
