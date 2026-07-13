"""A6 — instruction-text normalization parity.

Pure (no torch), so it runs anywhere. Proves training and inference feed T5 identical text:
training applies dataset `.lower()` + the collator's punctuation strip, then the conditioner
re-normalizes; inference now applies the same `normalize_instruction`. The two must agree.
"""
import re
import string

from ditar_encoder.encoders.text_normalization import NULL_INSTRUCTION, normalize_instruction

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def _training_pipeline(raw: str) -> str:
    """What training feeds the conditioner: dataset `.lower()` then collator punctuation strip."""
    return _PUNCT.sub("", raw.lower())


def test_basic_cases():
    assert normalize_instruction("Pick up the red block.") == "pick up the red block"
    assert normalize_instruction("open the drawer") == "open the drawer"
    assert normalize_instruction("  Close   the BOX!  ") == "close   the box"


def test_null_sentinel_and_empty():
    assert normalize_instruction("NULL") == "NULL"          # CFG/dropout sentinel preserved
    assert normalize_instruction("") == NULL_INSTRUCTION
    assert normalize_instruction(None) == NULL_INSTRUCTION
    assert normalize_instruction("...") == NULL_INSTRUCTION  # all-punctuation -> empty -> NULL


def test_idempotent():
    for s in ["pick up the cube", "NULL", "turn the faucet", "Pick up the RED block!"]:
        once = normalize_instruction(s)
        assert normalize_instruction(once) == once


def test_train_inference_parity():
    # The heart of A6: inference of raw text must equal the (conditioner-renormalized) text the
    # training pipeline produced. No casing/punctuation skew into T5.
    for raw in [
        "Pick up the RED block!",
        "Turn the faucet, slowly.",
        "open the drawer",
        "Push   the button?",
        "CLOSE THE BOX.",
    ]:
        train_then_norm = normalize_instruction(_training_pipeline(raw))
        infer_norm = normalize_instruction(raw)
        assert train_then_norm == infer_norm, raw


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"All {len(fns)} pure normalization tests passed.")
