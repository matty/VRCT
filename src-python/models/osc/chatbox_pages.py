"""Pages for chatbox messages that exceed VRChat's limit.

VRChat shows at most 144 characters of a chatbox message and silently drops
the rest. With the default format ("message\ntranslation") the part that gets
cut is the end of the translation - the part the listener needs. An utterance
of about 5 seconds is already enough to hit the limit.

Instead of letting VRChat cut the text, a long message is shown page by page.
Every page is rendered with the user's own message format and holds the next
part of the original together with the next part of each translation.

Pages are filled front to back. Transcription keeps sending longer versions
of the utterance that is still being spoken, and filling front to back means
a page that is full does not change when more text arrives: new text only
ever lands on the last page. (Balancing the pages instead would move every
page boundary each time the text grows.)

Where a page ends, in order of preference:
1. after the same number of whole sentences in the original and in every
   translation, so the lines of a page say the same thing;
2. otherwise each text gets a share of the page in proportion to its length
   and is cut at the last sentence end, clause end or space inside its share
   (anywhere for text without spaces, such as Japanese).

`ChatboxPlayer` (chatbox_player.py) decides when each page is shown.
"""

import re
from typing import Callable, List, Optional, Sequence, Tuple

CHATBOX_MAX_CHARS = 144
# How many pages ahead are laid out at a time.
CHATBOX_MAX_PAGES = 8
# Room kept for the " (12/15)" marker on the pages of a message that needs
# more than one.
_MARKER_RESERVE = 8

# Sentence end: ASCII terminators need a following space or the end of the
# text (so "3.5" and "a.m" stay whole), full-width ones do not.
_SENTENCE_END_RE = re.compile(
    r"(?:[.!?…]+[\"'’”)\]]*(?=\s|$)|[。！？．]+[」』）)\"'”]*)"
)
_CLAUSE_END_RE = re.compile(r"(?:[,;:]+(?=\s)|[、，；：]+)")
_SPACE_RE = re.compile(r"\s+")

Render = Callable[[List[str], str], str]


def chatboxLength(text: str) -> int:
    """Length as VRChat counts it (UTF-16 code units: an emoji counts as 2)."""
    return len(text.encode("utf-16-le")) // 2


def _renderParts(render: Render, parts: Sequence[str]) -> str:
    """Render a page from [message, *translations], leaving out empty translations."""
    return render([part for part in parts[1:] if part], parts[0])


def _prefixCut(text: str, share: int) -> int:
    """How many characters of `text` to put on a page that has room for
    `share` of them: up to the last sentence end, clause end or space."""
    if len(text) <= share:
        return len(text)
    if share <= 0:
        return 0
    # A sentence end is worth giving up more of the page for than a space is.
    for pattern, least in ((_SENTENCE_END_RE, 0.3), (_CLAUSE_END_RE, 0.4), (_SPACE_RE, 0.6)):
        low = int(share * least)
        ends = [m.end() for m in pattern.finditer(text, 0, share + 1) if low < m.end() <= share]
        if ends:
            return ends[-1]
    return share


def _sentenceEnds(text: str) -> List[int]:
    """Index just past each complete sentence of `text`."""
    return [m.end() for m in _SENTENCE_END_RE.finditer(text)]


def _takeAlignedSentences(
    texts: Sequence[str],
    active: Sequence[int],
    render: Render,
    limit: int,
) -> Optional[Tuple[str, List[int]]]:
    """A page made of the first k sentences of every text, for the largest k
    that fits - so the page ends where a sentence ends in the original and in
    each translation alike. None when that would leave the page mostly empty
    (the next sentence is too long to fit) or the texts have no sentence ends."""
    ends = [_sentenceEnds(texts[i]) for i in active]
    count = min(len(e) for e in ends)
    best: Optional[Tuple[str, List[int]]] = None
    for k in range(1, count + 1):
        used = [0] * len(texts)
        for i, e in zip(active, ends):
            used[i] = e[k - 1]
        parts = [texts[i][:used[i]].strip() for i in range(len(texts))]
        page = _renderParts(render, parts)
        if chatboxLength(page) > limit:
            break
        best = (page, used)
    if best is None or chatboxLength(best[0]) < limit // 3:
        return None
    return best


def _takeStreamPage(
    texts: Sequence[str],
    weights: Sequence[int],
    render: Render,
    limit: int,
) -> Tuple[str, List[int]]:
    """One page from the front of `texts` ([message, *translations]).

    Each text gets a share of the page in proportion to `weights` (the full
    length of that text), so the original and its translations advance
    together. Returns the rendered page and the characters used of each text.
    """
    active = [i for i, text in enumerate(texts) if text]
    whole = _renderParts(render, texts)
    if chatboxLength(whole) <= limit:
        return whole, [len(text) for text in texts]
    aligned = _takeAlignedSentences(texts, active, render, limit)
    if aligned is not None:
        return aligned
    overhead = chatboxLength(render(["x" for text in texts[1:] if text], "x" if texts[0] else "")) - len(active)
    budget = float(max(len(active), limit - overhead))
    for _ in range(40):
        used = [0] * len(texts)
        # A text that is short anyway goes on this page whole ("はい" with a
        # long translation must not become "は" and "い" on two pages). The
        # others share what is left.
        small = budget / (2 * len(active))
        whole_texts = [i for i in active if len(texts[i]) <= small]
        sharing = [i for i in active if i not in whole_texts]
        for i in whole_texts:
            used[i] = len(texts[i])
        room = budget - sum(used)
        sharing_weight = sum(max(1, weights[i]) for i in sharing)
        for i in sharing:
            share = max(1, int(room * max(1, weights[i]) / sharing_weight))
            used[i] = _prefixCut(texts[i], share)
        parts = [texts[i][:used[i]].strip() for i in range(len(texts))]
        page = _renderParts(render, parts)
        if chatboxLength(page) <= limit and any(used):
            return page, used
        budget *= 0.9
        if budget < len(active):
            break
    # The format leaves no room to share a page: one text at a time.
    first = active[0]
    used = [0] * len(texts)
    used[first] = 1
    while used[first] < len(texts[first]):
        parts = [texts[i][:used[i] + (1 if i == first else 0)] for i in range(len(texts))]
        if chatboxLength(_renderParts(render, parts)) > limit:
            break
        used[first] += 1
    parts = [texts[i][:used[i]] for i in range(len(texts))]
    return _renderParts(render, parts), used


def buildStreamPages(
    texts: Sequence[str],
    offsets: Sequence[int],
    completed: int,
    render: Render,
    limit: int = CHATBOX_MAX_CHARS,
    max_pages: int = CHATBOX_MAX_PAGES,
) -> List[Tuple[str, List[int]]]:
    """Pages for the part of a message that has not been shown yet.

    `texts` is [message, *translations]; `offsets` says how many characters
    of each have already been shown on the `completed` pages that came
    before ([0, ...] and 0 for a message nobody has seen yet). Returns
    (page, characters of each text on it) from the page that should be on
    screen now onwards, at most `max_pages` of them.

    A message that fits is a single page, exactly as `render` produced it.
    Pages of a longer message end with a " (3/5)" marker.
    """
    remaining = []
    skipped = []
    for text, offset in zip(texts, offsets):
        rest = text[offset:]
        stripped = rest.lstrip()
        skipped.append(len(rest) - len(stripped))
        remaining.append(stripped.rstrip())
    if not any(remaining):
        return []
    weights = [len(text.strip()) for text in texts]

    if completed == 0:
        whole = _renderParts(render, remaining)
        if chatboxLength(whole) <= limit:
            return [(whole, [len(text) - offset for text, offset in zip(texts, offsets)])]

    pages: List[Tuple[str, List[int]]] = []
    current = list(remaining)
    lead = list(skipped)
    while any(current) and len(pages) < max_pages:
        page, used = _takeStreamPage(current, weights, render, limit - _MARKER_RESERVE)
        consumed = []
        for i, count in enumerate(used):
            rest = current[i][count:]
            stripped = rest.lstrip()
            # Whitespace around a cut belongs to the page that ends there.
            consumed.append(lead[i] + count + (len(rest) - len(stripped)))
            current[i] = stripped
            lead[i] = 0
        pages.append((page, consumed))
    total = completed + len(pages)
    return [
        (f"{page} ({completed + index}/{total})", consumed)
        for index, (page, consumed) in enumerate(pages, start=1)
    ]
