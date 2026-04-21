"""Named Entity Recognition task classes: CoNLL-2003, FewNERD, WikiANN.

All produce single-turn conversations: user=text with instruction suffix,
assistant=newline-separated entity list.

Format (designed for 30-40% assistant/total ratio to avoid summary's sparse-loss failure):

  User:      {text}\n\nEntities:
  Assistant: - Barack Obama [PERSON]
             - Berlin [LOCATION]
             - NATO [ORGANIZATION]
"""

from datasets import load_dataset
from tasks.common import Task


# Shared tag mapping: collapse various dataset schemas to a stable 4-5 class label
_COARSE_TAGS = {
    # CoNLL-2003 BIO tags
    "B-PER": "PERSON", "I-PER": "PERSON",
    "B-ORG": "ORGANIZATION", "I-ORG": "ORGANIZATION",
    "B-LOC": "LOCATION", "I-LOC": "LOCATION",
    "B-MISC": "MISC", "I-MISC": "MISC",
    # WikiANN uses same PER/ORG/LOC in BIO
    # FewNERD has 8 coarse types; we keep most common:
    "person": "PERSON",
    "organization": "ORGANIZATION",
    "location": "LOCATION",
    "art": "MISC", "building": "LOCATION", "event": "MISC",
    "product": "MISC", "other": "MISC",
}


def _bio_to_spans(tokens, tags, id2label=None):
    """Convert BIO-tagged token list to list of (text, tag) spans.
    Accepts either string tags or int indices + id2label mapping.
    """
    spans = []
    cur_tokens = []
    cur_tag = None
    for tok, tag in zip(tokens, tags):
        if isinstance(tag, int) and id2label is not None:
            tag = id2label[tag]
        coarse = _COARSE_TAGS.get(tag, None)
        if tag == "O" or tag is None or coarse is None:
            if cur_tokens:
                spans.append((" ".join(cur_tokens), cur_tag))
                cur_tokens, cur_tag = [], None
            continue
        # B-* starts a new span, I-* continues (if same coarse type)
        if tag.startswith("B-") or cur_tag != coarse:
            if cur_tokens:
                spans.append((" ".join(cur_tokens), cur_tag))
            cur_tokens = [tok]
            cur_tag = coarse
        else:  # I-*
            cur_tokens.append(tok)
    if cur_tokens:
        spans.append((" ".join(cur_tokens), cur_tag))
    return spans


def _make_ner_conv(text, spans):
    """Build a conversation where user=text+cue and assistant=bulleted entities.
    If no spans, assistant says "None found." to keep assistant turn non-empty.
    """
    if spans:
        entity_lines = [f"- {name} [{tag}]" for name, tag in spans]
        assistant = "\n".join(entity_lines)
    else:
        assistant = "None found."
    return {
        "messages": [
            {"role": "user", "content": f"{text}\n\nEntities:"},
            {"role": "assistant", "content": assistant},
        ]
    }


class CoNLL2003(Task):
    """eriktks/conll2003 — 18k sentences, PER/ORG/LOC/MISC. Research license."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "validation", "test"), f"CoNLL split: {split}"
        self.ds = load_dataset("eriktks/conll2003", split=split).shuffle(seed=42)
        self.id2label = self.ds.features["ner_tags"].feature.names
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        text = " ".join(row["tokens"])
        spans = _bio_to_spans(row["tokens"], row["ner_tags"], self.id2label)
        return _make_ner_conv(text, spans)


class FewNERD(Task):
    """DFKI-SLT/few-nerd (supervised) — 188k sentences, 8 coarse types. CC-BY-SA 4.0."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "validation", "test"), f"FewNERD split: {split}"
        self.ds = load_dataset(
            "DFKI-SLT/few-nerd", "supervised", split=split
        ).shuffle(seed=42)
        self.id2label = self.ds.features["ner_tags"].feature.names
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        # FewNERD tags are single-token coarse labels (no BIO), so we group
        # consecutive identical-tag tokens manually.
        tokens = row["tokens"]
        tags = [self.id2label[t] for t in row["ner_tags"]]
        text = " ".join(tokens)
        spans = []
        cur_tokens, cur_tag = [], None
        for tok, tag in zip(tokens, tags):
            coarse = _COARSE_TAGS.get(tag, None)
            if coarse is None or tag == "O":
                if cur_tokens:
                    spans.append((" ".join(cur_tokens), cur_tag))
                    cur_tokens, cur_tag = [], None
                continue
            if cur_tag == coarse:
                cur_tokens.append(tok)
            else:
                if cur_tokens:
                    spans.append((" ".join(cur_tokens), cur_tag))
                cur_tokens = [tok]
                cur_tag = coarse
        if cur_tokens:
            spans.append((" ".join(cur_tokens), cur_tag))
        return _make_ner_conv(text, spans)


class WikiANNEn(Task):
    """unimelb-nlp/wikiann (English subset) — ~20k sentences, PER/ORG/LOC. CC-BY-SA."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "validation", "test"), f"WikiANN split: {split}"
        self.ds = load_dataset(
            "unimelb-nlp/wikiann", "en", split=split
        ).shuffle(seed=42)
        self.id2label = self.ds.features["ner_tags"].feature.names
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        text = " ".join(row["tokens"])
        spans = _bio_to_spans(row["tokens"], row["ner_tags"], self.id2label)
        return _make_ner_conv(text, spans)
