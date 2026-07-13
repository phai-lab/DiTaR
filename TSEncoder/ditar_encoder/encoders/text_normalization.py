"""Instruction-text normalization shared by the training data pipeline and the frozen text
conditioner, so the text fed to T5 is identical at train and inference time (audit A6).

Pure stdlib (no torch) so it is import-light and unit-testable in isolation.
"""
from __future__ import annotations

import re
import string
from typing import Optional

_PUNCTUATION_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]")

# Classifier-free-guidance / text-dropout sentinel; preserved verbatim (never lowercased),
# so its learned T5 embedding stays consistent between training and inference.
NULL_INSTRUCTION = "NULL"


def normalize_instruction(instruction: Optional[str]) -> str:
    """Lowercase, strip punctuation, and trim a task instruction.

    Empty/None maps to ``NULL``; the ``NULL`` sentinel is preserved as-is. The transform is
    idempotent on already-normalized training text (the datasets lowercase, the collator strips
    punctuation), and applying it at inference removes the casing/punctuation skew that shifted
    T5 conditioning off-distribution.
    """
    text = instruction.strip() if instruction else ""
    if not text or text == NULL_INSTRUCTION:
        return NULL_INSTRUCTION
    text = _PUNCTUATION_PATTERN.sub("", text.lower()).strip()
    return text or NULL_INSTRUCTION
