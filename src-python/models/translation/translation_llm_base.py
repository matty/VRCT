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


# --- 翻訳専用モデルのプロファイル -------------------------------------------
# Ollama / LM Studio 等で動かせる翻訳専用のローカルモデルは、学習時の固定の
# プロンプト形式から外れると精度が落ちる (システムプロンプトを持たない、
# 指示文が 1 つに決まっている)。モデル名で判別し、公式の形式で問い合わせる。
#
#   - Tencent Hunyuan-MT-7B / HY-MT1.5 (1.8B, 7B): WMT25 で 31 中 30 の言語対で
#     1 位。1.8B は GGUF で 1GB 前後になり、ゲーム中の GPU でも動かしやすい。
#     出典: https://huggingface.co/tencent/HY-MT1.5-1.8B
#           https://huggingface.co/tencent/Hunyuan-MT-7B
#   - Google TranslateGemma (4B/12B/27B, 55 言語)。
#     出典: https://ollama.com/library/translategemma

PROFILE_DEFAULT = "default"
PROFILE_HUNYUAN_MT = "hunyuan_mt"
PROFILE_HY_MT15 = "hy_mt15"  # HY-MT1.5 は会話の文脈を渡す公式テンプレートがある
PROFILE_TRANSLATEGEMMA = "translategemma"

_HY_MT15_RE = re.compile(r"hy[-_]?mt[-_]?1\.5|hunyuan[-_]?mt[-_]?1\.5")
_HUNYUAN_MT_RE = re.compile(r"hunyuan[-_]?mt|hy[-_]?mt")
_TRANSLATEGEMMA_RE = re.compile(r"translate[-_]?gemma")

# モデルカードの推奨値 (Hunyuan-MT-7B / HY-MT1.5 共通)
_HUNYUAN_SAMPLING = {"temperature": 0.7, "top_p": 0.6, "top_k": 20, "repetition_penalty": 1.05}

# Hunyuan の中国語テンプレートに入れる言語名
_ZH_LANGUAGE_NAMES = {
    "English": "英语", "Japanese": "日语", "Korean": "韩语",
    "Chinese Simplified": "简体中文", "Chinese Traditional": "繁体中文", "Cantonese": "粤语",
    "French": "法语", "German": "德语", "Spanish": "西班牙语", "Portuguese": "葡萄牙语",
    "Italian": "意大利语", "Russian": "俄语", "Arabic": "阿拉伯语", "Thai": "泰语",
    "Vietnamese": "越南语", "Indonesian": "印尼语", "Malay": "马来语", "Filipino": "菲律宾语",
    "Tagalog": "菲律宾语", "Turkish": "土耳其语", "Hindi": "印地语", "Polish": "波兰语",
    "Czech": "捷克语", "Dutch": "荷兰语", "Ukrainian": "乌克兰语", "Hebrew": "希伯来语",
    "Persian": "波斯语", "Urdu": "乌尔都语", "Bengali": "孟加拉语", "Tamil": "泰米尔语",
    "Kazakh": "哈萨克语", "Mongolian": "蒙古语", "Khmer": "高棉语", "Burmese": "缅甸语",
}

# TranslateGemma の言語名・コード。コードは Google の表から引き、
# モデルの表記 (zh-Hans/zh-Hant) と表に無いものだけここで補う。
_TRANSLATEGEMMA_OVERRIDES = {
    "Chinese Simplified": ("Chinese", "zh-Hans"),
    "Chinese Traditional": ("Chinese", "zh-Hant"),
    "Norwegian": ("Norwegian", "no"),
    "Persian": ("Persian", "fa"),
    "Persian/Farsi": ("Persian", "fa"),
    "Tagalog": ("Tagalog", "tl"),
}


def detectModelProfile(model_name: Optional[str]) -> str:
    name = (model_name or "").lower()
    if _TRANSLATEGEMMA_RE.search(name):
        return PROFILE_TRANSLATEGEMMA
    if _HY_MT15_RE.search(name):
        return PROFILE_HY_MT15
    if _HUNYUAN_MT_RE.search(name):
        return PROFILE_HUNYUAN_MT
    return PROFILE_DEFAULT


def _isChinese(language: str) -> bool:
    return language.startswith("Chinese") or language == "Cantonese"


def _translateGemmaLanguage(language: str) -> tuple:
    if language in _TRANSLATEGEMMA_OVERRIDES:
        return _TRANSLATEGEMMA_OVERRIDES[language]
    code = translation_lang.get("Google", {}).get("source", {}).get(language)
    return language, code or language


def buildHunyuanPrompt(text: str, input_lang: str, output_lang: str, context: str = "", contextual: bool = False) -> str:
    """Hunyuan-MT / HY-MT1.5 の公式テンプレート。

    中国語が絡む言語対は中国語の指示文、それ以外は英語の指示文を使う
    (モデルカードの指定)。HY-MT1.5 は文脈付きテンプレート (中国語) を持つ。
    """
    if contextual and context:
        target = _ZH_LANGUAGE_NAMES.get(output_lang, output_lang)
        return f"{context}\n参考上面的信息，把下面的文本翻译成{target}，注意不需要翻译上文，也不要额外解释：\n{text}"
    if _isChinese(input_lang) or _isChinese(output_lang):
        target = _ZH_LANGUAGE_NAMES.get(output_lang, output_lang)
        return f"将以下文本翻译为{target}，注意只需要输出翻译后的结果，不要额外解释：\n{text}"
    return f"Translate the following segment into {output_lang}, without additional explanation.\n\n{text}"


def buildTranslateGemmaPrompt(text: str, input_lang: str, output_lang: str) -> str:
    """TranslateGemma の公式テンプレート (1 つの user メッセージ、本文の前に空行 2 つ)。"""
    source_lang, source_code = _translateGemmaLanguage(input_lang)
    target_lang, target_code = _translateGemmaLanguage(output_lang)
    return (
        f"You are a professional {source_lang} ({source_code}) to {target_lang} ({target_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {source_lang} text "
        f"while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {target_lang} translation, without any additional explanations or commentary. "
        f"Please translate the following {source_lang} text into {target_lang}:\n"
        f"\n"
        f"\n"
        f"{text}"
    )


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

    def modelProfile(self) -> str:
        return detectModelProfile(getattr(self, "model", None))

    def buildMessages(self, text: str, input_lang: str, output_lang: str) -> list[dict]:
        profile = self.modelProfile()
        if profile in (PROFILE_HUNYUAN_MT, PROFILE_HY_MT15):
            # システムプロンプトを持たないモデル。指示と本文を 1 つの user メッセージで渡す
            context = self._historyLines() if profile == PROFILE_HY_MT15 else ""
            prompt = buildHunyuanPrompt(text, input_lang, output_lang, context=context, contextual=bool(context))
            return [{"role": "user", "content": prompt}]
        if profile == PROFILE_TRANSLATEGEMMA:
            # 公式テンプレートに文脈の欄は無い。混ぜると形式から外れるので渡さない
            return [{"role": "user", "content": buildTranslateGemmaPrompt(text, input_lang, output_lang)}]

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

    def _invokeKwargs(self, llm: Any) -> dict:
        """モデルカードが推奨するサンプリング設定を、呼び出しごとに渡す。

        汎用モデルには何も足さない (プロバイダ/モデル側の既定値が推奨値で、
        gpt-5 系のように temperature の変更自体を拒否するモデルもある)。
        """
        if self.modelProfile() not in (PROFILE_HUNYUAN_MT, PROFILE_HY_MT15):
            return {}
        p = _HUNYUAN_SAMPLING
        kind = type(llm).__name__
        if kind == "ChatOllama":
            return {"options": {
                "temperature": p["temperature"],
                "top_p": p["top_p"],
                "top_k": p["top_k"],
                "repeat_penalty": p["repetition_penalty"],
            }}
        if kind == "ChatOpenAI":
            # LM Studio / llama.cpp server / vLLM / OpenRouter は top_k と
            # repetition_penalty を OpenAI 互換 API の拡張パラメータとして受け付ける
            return {
                "temperature": p["temperature"],
                "top_p": p["top_p"],
                "extra_body": {"top_k": p["top_k"], "repetition_penalty": p["repetition_penalty"]},
            }
        return {}

    def _translateWith(self, llm: Any, text: str, input_lang: str, output_lang: str) -> str:
        messages = self.buildMessages(text, input_lang, output_lang)
        resp = llm.invoke(messages, **self._invokeKwargs(llm))
        return cleanTranslationOutput(extractContent(resp), text)
