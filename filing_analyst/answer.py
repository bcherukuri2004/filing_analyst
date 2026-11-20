"""Writing the answer, and checking it. The model is told to answer only from
the numbered passages and to cite them. Afterwards plain code verifies the
citations and that every number in the answer really appears in a passage
the answer cites, because a wrong number in a financial answer is worse than
no answer."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from .store import Hit

NOT_FOUND = "I could not find this in the filings I have."
MIN_PASSAGES_TO_TRY = 1

SYSTEM = f"""You answer questions about company annual reports (Form 10-K) using only the numbered passages you are given.

Rules:
1. Use only facts stated in the passages. Do not use outside knowledge.
2. After every sentence that states a fact, cite the passage numbers like [1] or [2][3].
3. Copy numbers exactly as written in the passage, including the units (for example "$383,285 million").
4. If the passages do not contain the answer, reply with exactly: {NOT_FOUND}
5. Be brief. No introduction and no closing remarks."""


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str | None = None):
        import anthropic

        self.model = model or os.environ.get("FILING_ANALYST_MODEL", "claude-sonnet-5-5")
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=700, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def build_prompt(question: str, hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(f"[{i}] {h.label}\n{h.text}")
    return "Passages:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}"


_CITE_RE = re.compile(r"\[(\d{1,2})\]")
# Numbers that need support: anything with a digit that is more than a single small number
# or a bare year. Comma grouped numbers, decimals, and percentages count.
_NUM_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


def _norm_num(s: str) -> str:
    return s.replace("$", "").replace(",", "").rstrip("%").rstrip(".")


def numbers_in(text: str) -> list[str]:
    out = []
    for m in _NUM_RE.findall(text):
        n = _norm_num(m)
        if not n:
            continue
        digits = n.replace(".", "")
        if len(digits) <= 1:  # "1", "2" are almost always list numbers or counts
            continue
        out.append(n)
    return out


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    cited: list[int] = field(default_factory=list)  # 1 based passage numbers the answer cites
    problems: list[str] = field(default_factory=list)
    refused: bool = False

    @property
    def verified(self) -> bool:
        return not self.problems


def verify(text: str, hits: list[Hit]) -> tuple[list[int], list[str]]:
    problems: list[str] = []
    cited = sorted({int(n) for n in _CITE_RE.findall(text)})
    bad = [n for n in cited if n < 1 or n > len(hits)]
    if bad:
        problems.append(f"cites passages that do not exist: {bad}")
    cited = [n for n in cited if 1 <= n <= len(hits)]
    if not cited:
        problems.append("answer has no citations")
        return cited, problems
    source = " ".join(hits[n - 1].text for n in cited)
    source_nums = {_norm_num(m) for m in _NUM_RE.findall(source)}
    # Compare the answer's text with the citation markers removed, so "[12]" is not read as a number.
    body = _CITE_RE.sub("", text)
    missing = [n for n in numbers_in(body) if n not in source_nums]
    if missing:
        problems.append("numbers not found in the cited passages: " + ", ".join(sorted(set(missing))))
    return cited, problems


def answer_question(question: str, hits: list[Hit], llm: LLM) -> Answer:
    if len(hits) < MIN_PASSAGES_TO_TRY:
        return Answer(question, NOT_FOUND, hits, refused=True)
    text = llm.complete(SYSTEM, build_prompt(question, hits))
    if text.strip().startswith(NOT_FOUND):
        return Answer(question, NOT_FOUND, hits, refused=True)
    cited, problems = verify(text, hits)
    return Answer(question, text, hits, cited, problems)
