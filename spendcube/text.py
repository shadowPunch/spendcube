"""Text normalisation shared by matching, de-duplication and classification."""

from __future__ import annotations

import re

# Legal-form and filler words dropped from supplier names before comparing them.
LEGAL_FORMS = frozenset(
    """PTY PTYLTD PTE LTD LTDA LIMITED CC INC INCORPORATED LLC LLP GMBH AG BV NV SA SAS SARL
    SPA SRL SL PLC CO CORP CORPORATION KG OY AB AS SE SAB CV THE AND""".split()
)

# OCR confuses these digits with letters; only applied inside word-like tokens.
_OCR_DIGIT_TO_LETTER = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})
_TOKEN = re.compile(r"[A-Z0-9]+")
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")
_NON_ALNUM = re.compile(r"[^A-Z0-9]+")
_DIGITS = re.compile(r"\d+")


def _is_text(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def supplier_norm(name) -> str:
    """Comparable supplier key: upper case, no punctuation, no legal form, no spaces.

    Removing all spaces merges OCR splits such as 'PRAH AVEN' with 'PRAHAVEN'.
    """
    if not _is_text(name):
        return ""
    text = name.upper().replace("&", " AND ").replace(".", "")
    tokens = [t for t in re.split(r"[^A-Z0-9]+", text) if t and t not in LEGAL_FORMS]
    return "".join(tokens)


def _fold_token(token: str) -> str:
    """Fix OCR digit/letter swaps in word-like tokens ('RE5IN' -> 'RESIN')."""
    if len(token) < 3:
        return token
    digits = sum(ch.isdigit() for ch in token)
    if digits == 0 or digits > 2 or (len(token) - digits) / len(token) < 0.6:
        return token
    if any(ch.isdigit() and ch not in "0158" for ch in token):
        return token
    return token.translate(_OCR_DIGIT_TO_LETTER)


def desc_norm(text) -> str:
    """Readable normalised description: upper case, '3,5' -> '3.5', OCR folded."""
    if not _is_text(text):
        return ""
    upper = _DECIMAL_COMMA.sub(".", text.upper())
    return _TOKEN.sub(lambda m: _fold_token(m.group(0)), upper)


def desc_key(text) -> str:
    """Compact description key: normalised text with every non-alphanumeric removed."""
    return _NON_ALNUM.sub("", desc_norm(text))


def numbers(text) -> tuple[str, ...]:
    """Digit groups in a normalised description, used to stop '1500MM' merging with '1800MM'."""
    return tuple(_DIGITS.findall(desc_norm(text).replace(".", "")))


def invoice_core(ref) -> str:
    """Last run of digits without leading zeros: 'INV/2023/759857' and '759857' agree."""
    if not _is_text(ref):
        return ""
    runs = _DIGITS.findall(ref)
    return runs[-1].lstrip("0") if runs else _NON_ALNUM.sub("", ref.upper())
