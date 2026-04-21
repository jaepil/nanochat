"""
Salesforce/xlam-function-calling-60k: 60k single-turn function-calling
examples. CC-BY-4.0. Requires attribution + APIGen paper citation.
https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k

Raw format:
  { "query": "<user query>",
    "tools": [...structured tool defs...],
    "answers": [{"name": "...", "arguments": {...}}, ...] }

We format each row as a single user/assistant turn:
  - system (as first user message): tool definitions (JSON serialized)
  - user: the query
  - assistant: JSON-serialized list of tool calls
"""

import json
from datasets import load_dataset
from tasks.common import Task


def _as_obj(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except (TypeError, ValueError):
            return x
    return x


class XlamFunctionCalling(Task):
    """Salesforce/xlam-function-calling-60k. 60k rows, single-turn."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "XlamFunctionCalling only has train split"
        self.ds = load_dataset(
            "Salesforce/xlam-function-calling-60k", split=split
        ).shuffle(seed=42)
        self.length = len(self.ds)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        query = (row.get("query") or "").strip()
        tools = _as_obj(row.get("tools"))
        answers = _as_obj(row.get("answers"))

        tool_spec = (
            "You have access to the following tools. Call one or more when "
            "useful. Respond with a JSON list of calls, each shaped as "
            "{\"name\": ..., \"arguments\": {...}}.\n\nTOOLS:\n"
            + json.dumps(tools, ensure_ascii=False, indent=2)
        )
        user_content = f"{tool_spec}\n\nUSER: {query}"
        assistant_content = json.dumps(answers, ensure_ascii=False)

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        return {"messages": messages}
