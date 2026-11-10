"""Laughter in chat ("www", "草", "ㅋㅋㅋ", "lol", "jajaja" ...) for the local MT models.

NLLB / M2M100 were trained on written sentences and have never seen chat
laughter. They do not leave it alone - they invent a sentence for it:

    www               -> I'm not sure.
    草                -> The grass.
    それなwww          -> That's the Internet.
    それはやばいwww     -> It's a f***ing world.
    ㅋㅋㅋ             -> I did.
    qué gracioso jajaja -> What a funny s***.
    lol (en -> ja)    -> ロル

(NLLB-200-distilled-600M, measured; see
docs/translation_laughter_2026-11-10.md.)

So laughter is taken out before the text goes to the model and put back in
the target language's own convention afterwards. A message that is nothing
but laughter never reaches the model.

Only CTranslate2 uses this. The web engines and LLMs were not measured and
generally know chat slang.
"""

import re
from typing import List, Tuple

# What may follow a laughter token: the end of the text, whitespace or
# punctuation. "wwwにアクセスする" and "笑う" are not laughter.
_END = r"(?=$|[\s、。！？!?,.…~〜)）」』\"'])"
# What may precede a token that is written as a word of its own.
_START = r"(?:^|(?<=[\s、。！？!?,.…~〜(（「『\"']))"

# Latin-script laughter that is safe anywhere between word boundaries.
_HAHA = _START + r"(?:a?(?:ha){2,}h?|(?:he){2,}h?|(?:hi){2,}|(?:ho){2,})" + _END


def _clauseEdge(token: str) -> str:
    """`token` only where it opens or closes a clause.

    For tokens that are also ordinary words or numbers ("lol" the game, 555):
    "that's funny lol" and "lol what" are laughter, "I play lol every day" is
    left to the model.
    """
    edge = r"[、。！？!?,.…~〜]"
    return (
        r"(?:(?:^|(?<=" + edge + r")|(?<=" + edge + r"\s))(?:" + token + r")" + _END
        + r"|" + _START + r"(?:" + token + r")(?=\s*(?:$|" + edge + r")))"
    )


_JAPANESE = (
    r"(?<=[^\x00-\x7F])[wｗ]{3,}"                       # いやwwwそれは
    r"|(?<=[^\x00-\x7F])[wｗ]+" + _END                  # まじかw
    + r"|" + _START + r"[wｗ]{2,}(?!\.[A-Za-z0-9])" + _END  # www (not www.example.com)
    + r"|[（(]\s*笑+\s*[)）]"                            # (笑)
    + r"|(?<![一-鿿])笑+" + _END                         # 楽しかった笑 (not 爆笑, 笑う)
    + r"|大草原" + _END
    + r"|草(?:生|は)え(?:る|た|ます|ました)?" + _END      # 草生える
    # 草 on its own or after a particle: それは草. Not 庭の草.
    + r"|(?:^|(?<=[はでてもにがとわねよかんまじー！!？?、。\swｗ]))草+(?=$|[\s、。！？!?,.…~〜)）」』wｗ])"
    + r"|(?:ワロタ|わろた|ワロス)" + _END
    + r"|" + _START + r"(?:あ?は{3,}|ア?ハ{3,}|あは+は|アハ+ハ)" + _END  # ははは, あはは
)

_KOREAN = (
    r"[ㅋㅎ]{2,}"                          # ㅋㅋㅋ, ㅎㅎ
    r"|(?<=[가-힣])[ㅋㅎ]" + _END           # 웃겨ㅋ
)

_CHINESE = (
    # 哈哈哈. Not the words 哈哈大笑, 笑嘻嘻, 乐呵呵.
    r"(?<![笑乐樂])[哈呵嘿嘻]{2,}(?![大地一笑])"
    r"|笑死(?:我了|了)?" + _END
    + r"|" + _clauseEdge(r"h{3,}|23{3,}")  # hhh, 2333
)

_ENGLISH = _clauseEdge(r"lo+l+|lmf?ao+|rofl|lul+|kek+")

_PATTERNS = {
    "Japanese": _JAPANESE,
    "Korean": _KOREAN,
    "Chinese Simplified": _CHINESE,
    "Chinese Traditional": _CHINESE,
    "English": _ENGLISH,
    "Spanish": _START + r"(?:(?:ja){2,}j?|(?:je){2,}j?|(?:ji){2,})" + _END,
    "Portuguese": _clauseEdge(r"k{3,}|(?:rs){2,}|(?:hua){2,}|(?:hue){2,}"),
    "Thai": _clauseEdge(r"5{3,}\+?"),
    "Russian": _START + r"(?:[ах]?(?:ха){2,}х?|(?:хи){2,}|(?:хе){2,})" + _END,
    "Indonesian": _START + r"(?:(?:wk){2,}w?)" + _END,
    "French": _clauseEdge(r"mdr+|ptdr+"),
}

_COMPILED = {
    # "haha" is understood (and typed) everywhere.
    language: re.compile(f"(?:{pattern}|{_HAHA})", re.IGNORECASE)
    for language, pattern in _PATTERNS.items()
}
_DEFAULT = re.compile(_HAHA, re.IGNORECASE)

# How each language writes laughter in chat. Anything else gets "haha".
_LAUGHTER = {
    "Japanese": "笑",
    "English": "lol",
    "Korean": "ㅋㅋㅋ",
    "Chinese Simplified": "哈哈哈",
    "Chinese Traditional": "哈哈哈",
    "Spanish": "jajaja",
    "Portuguese": "kkkk",
    "Thai": "555",
    "Russian": "ахаха",
    "Indonesian": "wkwk",
    "French": "mdr",
}
_DEFAULT_LAUGHTER = "haha"
# Written directly after the sentence, without a space.
_NO_SPACE_BEFORE_LAUGHTER = frozenset({"Japanese", "Chinese Simplified", "Chinese Traditional"})

_HAS_WORD = re.compile(r"\w")
_FULLWIDTH_END = re.compile(r"[、。」』】！），．：；？］｝]$")

# One piece of a message: (is_laughter, text). `text` is "" for laughter.
Piece = Tuple[bool, str]


def laughterFor(language: str) -> str:
    return _LAUGHTER.get(language, _DEFAULT_LAUGHTER)


def splitLaughter(message: str, source_language: str) -> List[Piece]:
    """Split `message` into text and laughter, in order.

    Laughter that follows laughter is merged, and text with nothing to
    translate between or around laughter ("！！", spaces) is dropped. A
    message without laughter comes back as a single text piece, unchanged.
    """
    pattern = _COMPILED.get(source_language, _DEFAULT)
    pieces: List[Piece] = []
    position = 0
    found = False
    for match in pattern.finditer(message):
        if match.end() == match.start():
            continue
        found = True
        _appendText(pieces, message[position:match.start()])
        if not pieces or not pieces[-1][0]:
            pieces.append((True, ""))
        position = match.end()
    if not found:
        return [(False, message)]
    _appendText(pieces, message[position:])
    return pieces


def _appendText(pieces: List[Piece], text: str) -> None:
    text = text.strip()
    if pieces and pieces[-1][0]:
        # "great, lol. see you": the full stop belonged to the laughter.
        text = text.lstrip("、。，．,.").lstrip()
    if text and _HAS_WORD.search(text):
        pieces.append((False, text))


def joinLaughter(pieces: List[Piece], translated: List[str], target_language: str) -> str:
    """Put a message back together in the target language.

    `translated` holds the translation of each text piece of `pieces`, in
    order.
    """
    laughter = laughterFor(target_language)
    separator = "" if target_language in _NO_SPACE_BEFORE_LAUGHTER else " "
    result = ""
    texts = iter(translated)
    for is_laughter, _ in pieces:
        if is_laughter:
            # "That's crazy. lol" reads wrong: laughter belongs to the sentence.
            if result.endswith(("。", ".")) and not result.endswith(("..", "。。")):
                result = result[:-1]
            result += (separator if result else "") + laughter
            continue
        text = next(texts, "").strip()
        if not text:
            continue
        if result and not _FULLWIDTH_END.search(result):
            result += " "
        result += text
    return result
