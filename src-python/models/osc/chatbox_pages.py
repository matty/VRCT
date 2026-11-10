"""Split a chatbox message that exceeds VRChat's limit into pages.

VRChat shows at most 144 characters of a chatbox message and silently drops
the rest. With the default format ("message\\ntranslation")
the part that gets cut is the end of the translation - the part the listener
needs. An utterance of about 5 seconds is already enough to hit the limit.

Instead of letting VRChat cut the text, the message and each translation are
split into the same number of parts, and every page is rendered with the
user's own message format, so each page still shows the original together
with its translation. When the original and all translations have the same
number of sentences, pages are cut on the same sentence boundaries so the
parts on a page correspond to each other.
"""

import re
from typing import Callable, List, Optional, Sequence

CHATBOX_MAX_CHARS = 144
# More pages than this would keep the chatbox busy for too long. A message
# that still does not fit is cut into fixed-size pieces instead (see
# `_hardSplit`), so nothing is ever lost.
CHATBOX_MAX_PAGES = 8

# Sentence end: ASCII terminators need a following space (so "3.5" and "a.m"
# stay whole), full-width ones do not.
_SENTENCE_END_RE = re.compile(
    r"(?:[.!?…]+[\"'’”)\]]*(?=\s)|[。！？．]+[」』）)\"'”]*)"
)
_CLAUSE_END_RE = re.compile(r"(?:[,;:]+(?=\s)|[、，；：]+)")
_SPACE_RE = re.compile(r"\s+")


def chatboxLength(text: str) -> int:
    """Length as VRChat counts it (UTF-16 code units: an emoji counts as 2)."""
    return len(text.encode("utf-16-le")) // 2


def _splitSentences(text: str) -> List[str]:
    sentences: List[str] = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(text):
        piece = text[start:m.end()].strip()
        if piece:
            sentences.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _joinSentences(sentences: Sequence[str]) -> str:
    result = ""
    for sentence in sentences:
        if result and not re.search(r"[。！？．」』）、，]$", result):
            result += " "
        result += sentence
    return result


def _bestCut(text: str, start: int, target: int) -> int:
    """Index near `target` to cut `text[start:]` at: after the end of a
    sentence if there is one nearby, else after a clause, else at a space,
    else exactly at `target` (text without spaces, e.g. Japanese)."""
    remaining = len(text) - start
    window = max(6, (target - start) // 3)
    low = max(start + 1, target - window)
    high = min(len(text) - 1, target + window)
    if low > high or remaining <= 1:
        return min(max(target, start + 1), len(text))
    for pattern in (_SENTENCE_END_RE, _CLAUSE_END_RE, _SPACE_RE):
        candidates = [m.end() for m in pattern.finditer(text, start) if low <= m.end() <= high]
        if candidates:
            return min(candidates, key=lambda index: abs(index - target))
    return min(max(target, low), high)


def splitText(text: str, parts: int) -> List[str]:
    """Split `text` into `parts` pieces of similar length at natural
    boundaries. Pieces can be empty when the text is too short to split."""
    text = text.strip()
    if parts <= 1:
        return [text]
    if len(text) < parts * 2:
        return [text] + [""] * (parts - 1)
    pieces: List[str] = []
    start = 0
    for index in range(parts - 1):
        target = start + round((len(text) - start) / (parts - index))
        cut = _bestCut(text, start, target)
        pieces.append(text[start:cut].strip())
        start = cut
    pieces.append(text[start:].strip())
    return pieces


def _splitAligned(texts: Sequence[str], parts: int) -> Optional[List[List[str]]]:
    """Split every text on the same sentence boundaries.

    Only possible when all texts have the same number of sentences (which is
    how a sentence-by-sentence translation usually comes back). Returns one
    list of `parts` pieces per text, or None.
    """
    sentence_lists = [_splitSentences(text) for text in texts]
    count = len(sentence_lists[0])
    if count < parts or any(len(sentences) != count for sentences in sentence_lists):
        return None
    # Combined length of sentence i across all texts: pages are balanced on
    # what actually has to fit on them.
    weights = [sum(len(sentences[i]) for sentences in sentence_lists) for i in range(count)]
    cumulative = [0]
    for weight in weights:
        cumulative.append(cumulative[-1] + weight)
    boundaries: List[int] = []
    previous = 0
    for part in range(1, parts):
        target = cumulative[-1] * part / parts
        # At least one sentence on this page and on each remaining page.
        candidates = range(previous + 1, count - (parts - part) + 1)
        previous = min(candidates, key=lambda index: abs(cumulative[index] - target))
        boundaries.append(previous)
    boundaries.append(count)
    result: List[List[str]] = []
    for sentences in sentence_lists:
        pieces = []
        begin = 0
        for end in boundaries:
            pieces.append(_joinSentences(sentences[begin:end]))
            begin = end
        result.append(pieces)
    return result


def _hardSplit(text: str, limit: int) -> List[str]:
    pages: List[str] = []
    current = ""
    for char in text:
        if chatboxLength(current + char) > limit:
            pages.append(current)
            current = ""
        current += char
    if current:
        pages.append(current)
    return [page.strip() for page in pages if page.strip()]


def buildChatboxPages(
    message: str,
    translation: Sequence[str],
    render: Callable[[List[str], str], str],
    limit: int = CHATBOX_MAX_CHARS,
    max_pages: int = CHATBOX_MAX_PAGES,
) -> List[str]:
    """Return the chatbox pages for one message.

    `render(translation, message)` formats one page (the controller passes
    its message formatter, so prefixes/suffixes/separators/order follow the
    user's settings). A message that fits is returned as a single page,
    exactly as `render` produced it. Pages of a split message end with a
    " (1/3)" style marker so readers know more is coming.
    """
    translation = list(translation)
    full = render(translation, message)
    if chatboxLength(full) <= limit:
        return [full]

    texts = [message] + translation
    for parts in range(2, max_pages + 1):
        marker_length = len(f" ({parts}/{parts})")
        for split in (_splitAligned(texts, parts), [splitText(text, parts) for text in texts]):
            if split is None:
                continue
            pages = []
            for page_index in range(parts):
                page_message = split[0][page_index]
                page_translation = [pieces[page_index] for pieces in split[1:] if pieces[page_index]]
                if not page_message and not page_translation:
                    continue
                pages.append(render(page_translation, page_message))
            if len(pages) > 1 and all(chatboxLength(page) <= limit - marker_length for page in pages):
                return [f"{page} ({index}/{len(pages)})" for index, page in enumerate(pages, start=1)]

    # The format itself (long prefixes/separators) or a huge message leaves
    # no way to fit whole parts: cut the rendered text into fixed-size pieces.
    marker_length = len(f" ({max_pages}/{max_pages})")
    pieces = _hardSplit(full, limit - marker_length)
    if len(pieces) <= 1:
        return pieces or [full]
    if len(pieces) > max_pages:
        pieces = pieces[:max_pages]
        pieces[-1] = pieces[-1][:-1].rstrip() + "…"
    return [f"{piece} ({index}/{len(pieces)})" for index, piece in enumerate(pieces, start=1)]
