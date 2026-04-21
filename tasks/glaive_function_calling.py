"""
glaiveai/glaive-function-calling-v2: ~113k multi-turn function-calling
conversations. Apache 2.0.
https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2

Raw format:
  { "system": "SYSTEM: ...function defs...",
    "chat": "USER: ...\nASSISTANT: ...<|endoftext|>USER: ...\n..." }

We parse `chat` into alternating user/assistant messages. FUNCTION RESPONSE
turns are mapped to the user role (the model treats them as incoming context).
"""

import re
from datasets import load_dataset
from tasks.common import Task


_ROLE_RE = re.compile(
    r"^\s*(USER|ASSISTANT|FUNCTION RESPONSE)\s*:\s*",
    re.MULTILINE,
)
_EOT_MARKERS = ("<|endoftext|>", "</s>")


def _parse_chat(chat: str):
    """Split the flat `chat` string into a list of (role, content) tuples."""
    # Normalize terminators out, we don't need them.
    for m in _EOT_MARKERS:
        chat = chat.replace(m, "")

    matches = list(_ROLE_RE.finditer(chat))
    if not matches:
        return []
    turns = []
    for i, m in enumerate(matches):
        tag = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chat)
        content = chat[start:end].strip()
        if not content:
            continue
        if tag == "USER" or tag == "FUNCTION RESPONSE":
            role = "user"
        else:
            role = "assistant"
        # Merge consecutive same-role turns (uncommon but keeps alternation clean)
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + "\n\n" + content)
        else:
            turns.append((role, content))
    return turns


class GlaiveFunctionCalling(Task):
    """glaiveai/glaive-function-calling-v2. 113k rows, multi-turn."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "GlaiveFunctionCalling only has train split"
        self.ds = load_dataset(
            "glaiveai/glaive-function-calling-v2", split=split
        ).shuffle(seed=42)
        self.length = len(self.ds)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        system = (row.get("system") or "").strip()
        # Strip redundant "SYSTEM:" prefix if present so it is not double-marked.
        if system.startswith("SYSTEM:"):
            system = system[len("SYSTEM:"):].strip()
        chat = row.get("chat") or ""
        turns = _parse_chat(chat)
        messages = []
        if system:
            # Prepend as the first user turn so the nanochat renderer (which
            # expects alternating user/assistant without a separate system
            # slot) can handle it uniformly. The training objective masks out
            # user tokens anyway, so this injection does not leak into loss.
            if turns and turns[0][0] == "user":
                turns[0] = ("user", system + "\n\n" + turns[0][1])
            else:
                turns.insert(0, ("user", system))
        for role, content in turns:
            messages.append({"role": role, "content": content})
        return {"messages": messages}
