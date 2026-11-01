"""Token and cost accounting.

The reason this exists: every other RAG demo says "it works". This one tells
you what the answer cost, per question, in dollars. Finance teams ask that
question immediately after the first demo, and having the number pre-computed
is the difference between "let me get back to you" and a signed PO.
"""

from __future__ import annotations

import re

from config import MODEL_PRICES

_TOKEN_CHARS = 4  # rough English chars-per-token; conservative for mixed CJK


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # CJK characters count heavier than latin characters.
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(text) - cjk
    return max(1, int(latin / _TOKEN_CHARS) + cjk)


def cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = MODEL_PRICES.get(model.lower(), {"input": 0.0, "output": 0.0})
    return (prompt_tokens / 1000.0) * price["input"] + (completion_tokens / 1000.0) * price["output"]
