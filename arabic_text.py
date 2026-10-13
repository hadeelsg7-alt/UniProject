"""Arabic text utilities: normalisation with an index map back to the original
text, tokenisation and cue matching that tolerates attached prefixes.

The index map lets every tool quote *exact* spans of the original review as
evidence, even though matching is done on normalised text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

_DIACRITICS = set(chr(c) for c in list(range(0x0610, 0x061B)) + list(range(0x064B, 0x0660)) + [0x0670] + list(range(0x06D6, 0x06EE)))
_TATWEEL = "ـ"
_CHAR_MAP = {
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "‏": " ", "‎": " ", "\n": " ", "\r": " ", "\t": " ",
}
# Attached proclitics allowed before a cue (و ف ب ل ك + optional ال)
_PREFIX = r"(?:و|ف|ب|ل|ك|وب|ول|وال|بال|فال|لل|ولل|ال)?"
_BOUNDARY_LEFT = r"(?:(?<=[\s\W])|^)"


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Return (normalised_text, index_map) where index_map[i] is the index in
    the original text of normalised character i."""
    chars: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text or ""):
        if ch in _DIACRITICS or ch == _TATWEEL:
            continue
        ch = _CHAR_MAP.get(ch, ch).lower()
        if ch == " " and chars and chars[-1] == " ":
            continue
        chars.append(ch)
        idx.append(i)
    # collapse elongation: runs of 3+ identical letters -> 1 (e.g. جداااا -> جدا)
    out: list[str] = []
    out_idx: list[int] = []
    j = 0
    while j < len(chars):
        k = j
        while k + 1 < len(chars) and chars[k + 1] == chars[j]:
            k += 1
        run = k - j + 1
        keep = 1 if (run >= 3 and chars[j].isalpha()) else run
        out.extend(chars[j : j + keep])
        out_idx.extend(idx[j : j + keep])
        j = k + 1
    return "".join(out), out_idx


def normalize(text: str) -> str:
    return normalize_with_map(text)[0].strip()


_TOKEN_RE = re.compile(r"[\w؀-ۿ]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize(text))


_BOUNDARY_RIGHT = r"(?=[\s\W]|$)"


@lru_cache(maxsize=4096)
def _cue_regex(cue_norm: str, whole_word: bool = False) -> re.Pattern:
    if not re.match(r"[\w؀-ۿ]", cue_norm[0]):
        return re.compile(re.escape(cue_norm))  # punctuation cue such as ؟
    right = _BOUNDARY_RIGHT if whole_word else ""
    return re.compile(_BOUNDARY_LEFT + _PREFIX + re.escape(cue_norm) + right)


@dataclass
class CueMatch:
    cue: str
    start: int  # index in ORIGINAL text
    end: int  # exclusive index in ORIGINAL text
    norm_start: int
    norm_end: int


def find_cues(
    text: str,
    cues: list[str],
    norm_cache: tuple[str, list[int]] | None = None,
    whole_word: bool = False,
) -> list[CueMatch]:
    """Find all occurrences of the (already normalised) cues in `text`.
    Overlapping matches are resolved in favour of the longest cue."""
    norm, idx = norm_cache or normalize_with_map(text)
    found: list[CueMatch] = []
    for cue in cues:
        for m in _cue_regex(cue, whole_word).finditer(norm):
            s, e = m.start(), m.end()
            if e <= s:
                continue
            found.append(CueMatch(cue, idx[s], idx[e - 1] + 1, s, e))
    found.sort(key=lambda c: (c.norm_start, -(c.norm_end - c.norm_start)))
    result: list[CueMatch] = []
    last_end = -1
    for c in found:
        if c.norm_start >= last_end:
            result.append(c)
            last_end = c.norm_end
        elif c.norm_end - c.norm_start > result[-1].norm_end - result[-1].norm_start and c.norm_start == result[-1].norm_start:
            result[-1] = c
            last_end = c.norm_end
    return result


def has_cue(text: str, cues: list[str], whole_word: bool = False) -> bool:
    return bool(find_cues(text, cues, whole_word=whole_word))


def word_count(text: str) -> int:
    return len(tokens(text))


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def containment(part: str, whole: str) -> float:
    """Share of the tokens of `part` that appear in `whole`."""
    tp, tw = tokens(part), set(tokens(whole))
    if not tp:
        return 0.0
    return sum(1 for t in tp if t in tw) / len(tp)
