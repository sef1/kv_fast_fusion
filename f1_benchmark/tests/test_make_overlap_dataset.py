"""The overlap prompt set is only useful if every shared document starts at the SAME templated token
offset — that is what lines its blocks up across requests. Checked with a word-level tokenizer and a
chat template that adds a role prefix, so the padding loop has to account for the template. CPU only.
"""

import re

from f1_benchmark import make_overlap_dataset as mod


class _WordTok:
    """One token per word or newline; a chat template with a fixed role prefix."""

    def __init__(self):
        self.vocab, self.inv = {}, {}

    def _split(self, text):
        return re.findall(r"\n| ?[^\s]+", text)

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for t in self._split(text):
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
                self.inv[self.vocab[t]] = t
            ids.append(self.vocab[t])
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids):
        return "".join(self.inv[i] for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return "<|user|>\n" + msgs[0]["content"] + "\n<|assistant|>\n"


def test_every_document_starts_at_the_target_offset():
    tok = _WordTok()
    doc = mod.build_document(tok, [("q", " ".join(f"tok{i}" for i in range(300)))], 64)
    offsets = set()
    for i in range(5):
        core = "background " + " ".join(f"w{i}_{j}" for j in range(10 + 7 * i))
        prompt, _n = mod.build_prompt(tok, core, doc, "do it", 128, False)
        assert prompt is not None, f"prompt {i} could not be aligned"
        text = tok.apply_chat_template([{"role": "user", "content": prompt}])
        offsets.add(len(tok(text[:text.index(doc)]).input_ids))
    assert offsets == {128}


def test_a_document_is_cut_at_exactly_its_token_budget():
    tok = _WordTok()
    doc = mod.build_document(tok, [("q", " ".join(f"a{i}" for i in range(500)))], 100)
    assert len(tok(doc).input_ids) == 100


def test_a_core_longer_than_the_target_is_shrunk_to_fit():
    tok = _WordTok()
    doc = mod.build_document(tok, [("q", "d " * 200)], 32)
    core = " ".join(f"long{j}" for j in range(400))

    prompt, n = mod.build_prompt(tok, core, doc, "t", 64, False)

    assert prompt is not None and n == 64
