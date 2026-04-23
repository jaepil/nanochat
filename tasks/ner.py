"""Named Entity Recognition task classes: CoNLL-2003, FewNERD, WikiNeural.

All produce single-turn conversations: user=text with instruction suffix,
assistant=newline-separated entity list.

Format (designed for 30-40% assistant/total ratio to avoid summary's sparse-loss failure):

  User:      {text}\n\nEntities:
  Assistant: - Barack Obama [PERSON]
             - Berlin [LOCATION]
             - NATO [ORGANIZATION]

Dataset sources (parquet-only, since load-script datasets are no longer
supported by the datasets library):

  CoNLL-2003  -> tomaarsen/conll2003      (9-tag BIO, 14k train)
  FewNERD     -> DFKI-SLT/few-nerd        (9 coarse tags, 131k train, supervised config)
  WikiNeural  -> Babelscape/wikineural    (9-tag BIO, 109k train_en, replacement for WikiANN)
"""

from datasets import load_dataset
from tasks.common import Task


_COARSE_TAGS = {
    "B-PER": "PERSON", "I-PER": "PERSON",
    "B-PERSON": "PERSON", "I-PERSON": "PERSON",
    "B-ORG": "ORGANIZATION", "I-ORG": "ORGANIZATION",
    "B-LOC": "LOCATION", "I-LOC": "LOCATION",
    "B-MISC": "MISC", "I-MISC": "MISC",
    "person": "PERSON",
    "organization": "ORGANIZATION",
    "location": "LOCATION",
    "art": "MISC", "building": "LOCATION", "event": "MISC",
    "product": "MISC", "other": "MISC",
}


_WIKINEURAL_LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]


def _bio_to_spans(tokens, tags, id2label=None):
    """Convert BIO-tagged token list to list of (text, coarse_tag) spans.
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
        if tag.startswith("B-") or cur_tag != coarse:
            if cur_tokens:
                spans.append((" ".join(cur_tokens), cur_tag))
            cur_tokens = [tok]
            cur_tag = coarse
        else:
            cur_tokens.append(tok)
    if cur_tokens:
        spans.append((" ".join(cur_tokens), cur_tag))
    return spans


_TAG_PHRASE = {
    "PERSON": "a person",
    "ORGANIZATION": "an organization",
    "LOCATION": "a location",
    "MISC": "a miscellaneous entity",
}


def _make_ner_conv(text, spans):
    """Build a conversation in NATURAL LANGUAGE format (v5+).

    Format (autoregressive-friendly — escapes the BIO-list loop trap that
    plagued v1-v4):

      User:      {text}\n\nList the entities.
      Assistant: "Barack Obama" is a person. "Hawaii" is a location. \
                 "United States" is a location.

    Empty-entity case: "No entities found."
    """
    if spans:
        sentences = [f'"{name}" is {_TAG_PHRASE.get(tag, "an entity")}.'
                     for name, tag in spans]
        assistant = " ".join(sentences)
    else:
        assistant = "No entities found."
    return {
        "messages": [
            {"role": "user", "content": f"{text}\n\nList the entities."},
            {"role": "assistant", "content": assistant},
        ]
    }


class CoNLL2003(Task):
    """tomaarsen/conll2003 — 14k train sentences, PER/ORG/LOC/MISC (BIO)."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "validation", "test"), f"CoNLL split: {split}"
        self.ds = load_dataset("tomaarsen/conll2003", split=split).shuffle(seed=42)
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
    """DFKI-SLT/few-nerd (supervised) — 131k train sentences, 8 coarse types."""

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


class WikiNeuralEn(Task):
    """Babelscape/wikineural (English subset) — 109k train_en sentences, PER/ORG/LOC/MISC.

    Replaces the old `unimelb-nlp/wikiann` (script-based, no longer loadable).
    The ner_tags feature is a raw int64 list without a names mapping, so the
    BIO label schema is hardcoded per the Babelscape dataset card.
    """

    _SPLIT_MAP = {
        "train": "train_en",
        "validation": "val_en",
        "test": "test_en",
    }

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in self._SPLIT_MAP, f"WikiNeural split: {split}"
        self.ds = load_dataset(
            "Babelscape/wikineural", split=self._SPLIT_MAP[split]
        ).shuffle(seed=42)
        self.id2label = _WIKINEURAL_LABELS
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        text = " ".join(row["tokens"])
        spans = _bio_to_spans(row["tokens"], row["ner_tags"], self.id2label)
        return _make_ner_conv(text, spans)


WikiANNEn = WikiNeuralEn
