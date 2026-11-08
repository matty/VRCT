"""LLM 翻訳クライアント (Plamo/Gemini/OpenAI/Groq/OpenRouter/LM Studio/Ollama/
OpenAI 互換) の共通処理。

以前は各クライアントが translate() 内で「プロンプト整形 → 履歴注入 →
invoke → 応答の連結」をそれぞれコピーで持っていた (8 か所)。プロンプトの
改善や出力の後処理を 1 か所で直せるよう、ここへ集約する。

精度・速度のための主な変更点:
  - 履歴を「参照専用」と明示し、タグで囲んで本文と区別する。LLM が履歴まで
    翻訳して出力したり、翻訳対象の文に「返答」したりするのを防ぐ。
  - 履歴の発話者を mic/chat → "Me"、speaker/ocr → "Other" として渡す
    (生の "mic"/"speaker" ではモデルが意味を取れない)。
  - 出力から推論モデルの <think>...</think>、全体を囲む引用符、
    "Translation:" のような前置きを取り除く。
"""

from datetime import datetime
import re
from typing import Any, Optional

try:
    from .translation_languages import translation_lang
    from .translation_utils import loadTranslatePromptConfig
except Exception:
    import sys
    from os import path as os_path
    sys.path.append(os_path.dirname(os_path.abspath(__file__)))
    from translation_languages import translation_lang
    from translation_utils import loadTranslatePromptConfig


DEFAULT_HISTORY_CFG: dict = {
    "use_history": False,
    "sources": [],
    "max_messages": 0,
    "max_chars": 0,
    "header_template": "",
    "item_template": "{speaker}: {text}",
}

# 履歴の source をモデルが理解できる発話者ラベルへ変換する。
# mic/chat は自分の発言、speaker (相手の音声) と ocr (相手のチャット吹き出し)
# は相手の発言。
_SPEAKER_LABELS = {
    "mic": "Me",
    "chat": "Me",
    "speaker": "Other",
    "ocr": "Other",
}

# 応答全体を囲んでいたら外す引用符の組。
_QUOTE_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),  # “ ”
    ("‘", "’"),  # ‘ ’
    ("「", "」"),  # 「 」
    ("『", "』"),  # 『 』
    ("«", "»"),  # « »
    ("`", "`"),
)

_THINK_BLOCK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"</(think|thinking|reasoning)>", re.IGNORECASE)
_LABEL_RE = re.compile(r"^(?:translation|translated text|output)\s*[:：]\s*", re.IGNORECASE)


def extractContent(resp: Any) -> str:
    """LangChain の応答 (AIMessage) からテキストを取り出す。

    content は str のほか、マルチパートの list (str / {"type": "text", "text": ...}
    / {"content": ...}) で返るプロバイダがある。
    """
    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content
    text = ""
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                text += part
            elif isinstance(part, dict):
                # 推論過程 ({"type": "thinking"/"reasoning", ...}) は翻訳ではないので捨てる
                if part.get("type") in ("thinking", "reasoning"):
                    continue
                if isinstance(part.get("text"), str):
                    text += part["text"]
                elif isinstance(part.get("content"), str):
                    text += part["content"]
    return text


def cleanTranslationOutput(content: str, original: str = "") -> str:
    """LLM の応答から翻訳文以外の混入物を取り除く。

    original は翻訳元の文。元の文自体が引用符で囲まれている・"Translation:" で
    始まる場合は、それも訳文の一部なので外さない。
    """
    text = content or ""
    # 推論モデル (qwen3, deepseek-r1 等) は思考過程を <think> タグで出力する。
    # 閉じタグだけ残る (開始タグがテンプレート側で付与される) モデルもある。
    text = _THINK_BLOCK_RE.sub("", text)
    close = None
    for close in _THINK_CLOSE_RE.finditer(text):
        pass
    if close is not None:
        text = text[close.end():]
    text = text.strip()

    original = (original or "").strip()

    if not _LABEL_RE.match(original):
        text = _LABEL_RE.sub("", text, count=1).strip()

    for opening, closing in _QUOTE_PAIRS:
        if (
            len(text) >= 2
            and text.startswith(opening)
            and text.endswith(closing)
            and not original.startswith(opening)
            # 内側に同じ引用符があるなら全体を囲んでいるとは限らない ("a" and "b")
            and opening not in text[1:-1]
            and closing not in text[1:-1]
        ):
            text = text[1:-1].strip()
            break
    return text


class LLMTranslationMixin:
    """LLM 翻訳クライアントの共通部分。

    サブクラスは __init__ で `_initPrompt()` を呼び、translate() で
    `_translateWith(llm, ...)` を呼ぶ。
    """

    prompt_template: str
    history_cfg: dict
    supported_languages: list
    _context_history: list

    def _initPrompt(self, root_path: Optional[str], prompt_filename: str, engine_key: str) -> None:
        prompt_config = loadTranslatePromptConfig(root_path, prompt_filename)
        # {supported_languages} を使う独自プロンプト (ユーザーが編集した yml) の
        # ために値は保持するが、既定のプロンプトには入れない。毎リクエスト
        # 数百トークンを消費するうえ翻訳精度には寄与しない。
        self.supported_languages = list(translation_lang[engine_key]["source"].keys())
        self.prompt_template = prompt_config["system_prompt"]
        self.history_cfg = {**DEFAULT_HISTORY_CFG, **(prompt_config.get("history") or {})}
        self._context_history = []

    def setContextHistory(self, history_items: list[dict]) -> None:
        """Set recent conversation history for prompt injection.

        Each item should be a dict containing:
        - source: "chat" | "mic" | "speaker" | "ocr"
        - text: message string
        - timestamp: ISO format datetime string
        """
        self._context_history = list(history_items or [])

    def _historyLines(self) -> str:
        """設定 (yml の history) に従って整形した直近の会話。無効・空なら ""。"""
        cfg = self.history_cfg
        if not cfg.get("use_history"):
            return ""
        allowed_sources = set(cfg.get("sources") or [])
        max_messages = int(cfg.get("max_messages") or 0)
        max_chars = int(cfg.get("max_chars") or 0)
        item_tmpl = cfg.get("item_template") or DEFAULT_HISTORY_CFG["item_template"]

        filtered = [h for h in self._context_history if h.get("source") in allowed_sources]
        recent = filtered[-max_messages:] if max_messages > 0 else filtered

        formatted_items = []
        for h in recent:
            timestamp_str = ""
            if h.get("timestamp"):
                try:
                    timestamp_str = datetime.fromisoformat(h["timestamp"]).strftime("%H:%M")
                except Exception:
                    timestamp_str = ""
            source = h.get("source", "")
            formatted_items.append(
                item_tmpl.format(
                    timestamp=timestamp_str,
                    source=source,
                    speaker=_SPEAKER_LABELS.get(source, "Other"),
                    text=h.get("text", ""),
                )
            )
        history_blob = "\n".join(formatted_items).strip()
        if max_chars and len(history_blob) > max_chars:
            # 古い側を切り捨てる。行の途中から始まらないよう次の改行まで進める
            history_blob = history_blob[-max_chars:]
            newline = history_blob.find("\n")
            if 0 <= newline < len(history_blob) - 1:
                history_blob = history_blob[newline + 1:]
        return history_blob

    def _formatHistory(self) -> str:
        history_blob = self._historyLines()
        if not history_blob:
            # 空の履歴見出しを送ってもトークンを使うだけ
            return ""
        header_tmpl = self.history_cfg.get("header_template") or "{history}"
        max_messages = int(self.history_cfg.get("max_messages") or 0)
        return header_tmpl.format(max_messages=max_messages, history=history_blob).strip()

    def buildMessages(self, text: str, input_lang: str, output_lang: str) -> list[dict]:
        system_prompt = self.prompt_template.format(
            supported_languages=self.supported_languages,
            input_lang=input_lang,
            output_lang=output_lang,
        ).strip()
        # 指示 (固定) → 履歴 (可変) の順に並べる。先頭が毎回同じになるので、
        # プロバイダのプロンプトキャッシュやローカル LLM の KV キャッシュが効く。
        history = self._formatHistory()
        if history:
            system_prompt = f"{system_prompt}\n\n{history}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

    def _translateWith(self, llm: Any, text: str, input_lang: str, output_lang: str) -> str:
        messages = self.buildMessages(text, input_lang, output_lang)
        resp = llm.invoke(messages)
        return cleanTranslationOutput(extractContent(resp), text)
