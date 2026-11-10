import re
from collections import OrderedDict
from os import path as os_path
from deepl import DeepLClient

try:
    from .translation_languages import translation_lang
    from .translation_utils import ctranslate2_weights
    from .translation_providers import TRANSLATION_PROVIDER_REGISTRY
    from .translation_laughter import joinLaughter, splitLaughter
except Exception:
    import sys
    sys.path.append(os_path.dirname(os_path.dirname(os_path.dirname(os_path.abspath(__file__)))))
    from translation_languages import translation_lang
    from translation_utils import ctranslate2_weights
    from translation_providers import TRANSLATION_PROVIDER_REGISTRY
    from translation_laughter import joinLaughter, splitLaughter

from utils import errorLogging, getBestComputeType

try:
    # GPL の translators から、MIT 互換の自前実装に置き換えた
    # (translator_engines)。Bing の認証情報パースはそちらに内蔵済み。
    # Rust 製のビルド済み拡張 (src-python/translator_engines.pyd) を同梱している。
    # ライセンスにより VRCT 以外への流用は不可 (translator_engines.LICENSE.txt)。
    # translators と違い import 時に通信しないので、オフライン起動でも
    # ENABLE_TRANSLATORS が False に落ちない。
    from translator_engines import translate_text as other_web_Translator
    ENABLE_TRANSLATORS = True
except Exception:
    other_web_Translator = None  # type: ignore
    ENABLE_TRANSLATORS = False

# Google/Bing/Papago の HTTP タイムアウト。応答が返らないと呼び出し元の
# スレッドが止まる。VRCT のパイプラインは単一スレッドなので、翻訳のハングは
# 文字起こしごと停止させ、最終的に watchdog が発火する。
# NOTE: これは 1 リクエストあたりの値。Bing は認証ページの取得と翻訳、
# さらにトークン失効時の 1 回のリトライで最大 4 リクエストを発行しうる。
_WEB_TRANSLATOR_TIMEOUT_SECONDS = 10

import warnings
from threading import Lock, RLock
from typing import Any, Dict, Hashable, List, Optional, Tuple

warnings.filterwarnings("ignore")

# 結果をキャッシュするエンジン。入力 (言語ペア + 原文) だけで出力が決まるものに
# 限る。LLM 系と DeepL は会話履歴 (context) で訳が変わるため対象外。
# VRChat の会話は「こんにちは」「ありがとう」「www」のような短い定型句が
# 繰り返し出るので、ヒットすればネットワーク往復/デコードを丸ごと省ける。
_CACHEABLE_ENGINES = frozenset({"Google", "Bing", "Papago", "CTranslate2"})
_TRANSLATION_CACHE_MAX_ITEMS = 512

# DeepL の context に渡す直近の会話の上限。context は課金対象外。
_DEEPL_CONTEXT_MAX_MESSAGES = 5
_DEEPL_CONTEXT_MAX_CHARS = 1000

# CTranslate2 の出力トークン数の上限 = 入力トークン数 × 係数 + 余白。
# NLLB/M2M100 は短い入力で同じ句を繰り返し続ける幻覚を起こすことがあり、
# 既定 (256 トークン) まで生成し続けると数秒かかったうえに誤訳になる。
# 正常な訳がこの上限に届くことはまずない (日英間でも概ね 2 倍以内)。
_CT2_DECODING_LENGTH_RATIO = 3
_CT2_DECODING_LENGTH_MARGIN = 16
_CT2_DECODING_LENGTH_MAX = 256

# これより短い文 (トークン数、言語タグと </s> の 2 つを含む。"Hi!" = 4、
# "はじめまして！" = 6、"See you tomorrow, okay?" = 8) は前後の文とまとめて
# 訳す。「はじめまして！」のような短い挨拶を単独で訳すと文脈が無く誤訳しやすい
# (実測: NLLB-600M が "You're welcome!" と訳した。まとめれば "Hello!")。
_CT2_MIN_SEGMENT_TOKENS = 8

# --- CTranslate2 用の文分割 -------------------------------------------------
# NLLB/M2M100 は 1 文単位で学習されており、複数文をまとめて渡すと後半の文を
# 落としたり混ぜたりしやすい。文に分けて 1 回のバッチで訳し、つなぎ直す。
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "approx", "vol", "fig", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec",
})
# 文末記号 (と閉じ括弧・引用符) の直後。全角の文末記号は空白なしでも区切る。
_SENTENCE_END_RE = re.compile(
    r"(?:[.!?…]+[\"'’”)\]]*(?=\s)|[。！？．]+[」』）)\"'”]*)"
)
# 全角の句読点・閉じ括弧で終わる訳文の後ろには空白を入れずにつなぐ (日本語・
# 中国語の文は空白で区切らない)。NLLB は日本語の訳文に「。」を付けないことが
# 多く、その場合に空白なしでつなぐと文の境目が消えるので、句読点のときだけ。
_NO_SPACE_JOIN_RE = re.compile(
    "[、。」』】〕〗〙〛〞〟"
    "！），．：；？］｝｠]$"
)


def splitSentences(text: str) -> List[str]:
    """CTranslate2 に渡すために文へ分ける。分けられなければ [text]。"""
    segments: List[str] = []
    for line in text.splitlines():
        start = 0
        for m in _SENTENCE_END_RE.finditer(line):
            end = m.end()
            piece = line[start:end]
            following = line[end:].lstrip()[:1]
            if following.isascii() and following.islower():
                # '"Really?" she said.' のように小文字で続くなら文の途中
                continue
            if line[m.start()] == ".":
                # "Mr. Smith" "U.S. troops" "J. Smith" のような略語・頭文字では区切らない
                # ("3.5" は "." の直後が空白でないので、そもそも一致しない)
                word = re.findall(r"([A-Za-z.]+)\.$", piece.rstrip("\"')]’”"))
                if word and (word[0].lower() in _ABBREVIATIONS or len(word[0]) == 1 or "." in word[0]):
                    continue
            _appendSegment(segments, piece)
            start = end
        _appendSegment(segments, line[start:])
    return segments or [text]


def _appendSegment(segments: List[str], piece: str) -> None:
    piece = piece.strip()
    if not piece:
        return
    # 文字を含まない断片 (絵文字・記号だけ) を単独で訳させると、モデルが
    # 存在しない文を作ることがある。直前の文にくっつける。
    if segments and not re.search(r"\w", piece):
        segments[-1] = f"{segments[-1]} {piece}"
    else:
        segments.append(piece)


def joinSentences(sentences: List[str]) -> str:
    result = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if result and not _NO_SPACE_JOIN_RE.search(result):
            result += " "
        result += sentence
    return result



def mergeShortSentences(sentences: List[str], token_counts: List[int], min_tokens: int = _CT2_MIN_SEGMENT_TOKENS) -> List[str]:
    """短い文を後ろの文と (最後なら前の文と) まとめる。"""
    chunks: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0
    for sentence, count in zip(sentences, token_counts):
        current.append(sentence)
        current_tokens += count
        if current_tokens >= min_tokens:
            chunks.append(current)
            current, current_tokens = [], 0
    if current:
        if chunks:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return [joinSentences(chunk) for chunk in chunks]

class _TranslationCache:
    """スレッドセーフな LRU。getInputTranslate はターゲット言語ごとに並列で呼ぶ。"""

    def __init__(self, max_items: int = _TRANSLATION_CACHE_MAX_ITEMS) -> None:
        self._max_items = max_items
        self._items: "OrderedDict[Hashable, str]" = OrderedDict()
        self._lock = Lock()

    def get(self, key: Hashable) -> Optional[str]:
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key: Hashable, value: str) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class UnsupportedLanguageError(Exception):
    """Raised when a language is not supported by the selected translator.

    Distinct from a translator backend failure (rate limit, network,
    authentication, provider error) so callers do not disable the
    translation engine just because the user picked a language it
    doesn't cover.
    """
    pass

try:
    import ctranslate2  # noqa: F401
except Exception:
    ctranslate2 = None  # type: ignore

try:
    import transformers  # noqa: F401
except Exception:
    transformers = None  # type: ignore

# 各プロバイダの LLM クライアント SDK は起動時にまとめて import する。
# try/except は Python パッケージ vs スクリプト実行の両方で動かすため
# (`from .module` と `from module` の両パターン)。PyInstaller の import
# scanner が静的に検出できるよう `from ... import Class` の形を維持する。
try:
    from .translation_plamo import PlamoClient
except Exception:
    from translation_plamo import PlamoClient

try:
    from .translation_gemini import GeminiClient
except Exception:
    from translation_gemini import GeminiClient

try:
    from .translation_openai import OpenAIClient
except Exception:
    from translation_openai import OpenAIClient

try:
    from .translation_openai_compatible import OpenAICompatibleClient
except Exception:
    from translation_openai_compatible import OpenAICompatibleClient

try:
    from .translation_groq import GroqClient
except Exception:
    from translation_groq import GroqClient

try:
    from .translation_openrouter import OpenRouterClient
except Exception:
    from translation_openrouter import OpenRouterClient

try:
    from .translation_lmstudio import LMStudioClient
except Exception:
    from translation_lmstudio import LMStudioClient

try:
    from .translation_ollama import OllamaClient
except Exception:
    from translation_ollama import OllamaClient


class Translator:
    """High-level translator facade.

    This class wraps multiple backends (DeepL API, Google, Bing, Papago,
    and CTranslate2 local models). Optional dependencies may be unavailable at
    runtime; methods degrade gracefully and return False or an empty string on
    failure (kept compatible with existing behavior).
    """

    def __init__(self) -> None:
        self.is_enable_translators = ENABLE_TRANSLATORS
        self.deepl_client: Optional[DeepLClient] = None
        # TRANSLATION_PROVIDER_REGISTRY に登録されたエンジン
        # (Plamo/Gemini/OpenAI/Groq/OpenRouter、フェーズ3項目17) の
        # クライアントは、ここに一元管理する。エンジンごとの専用フィールド
        # (self.plamo_client 等、旧実装) の代わりに、レジストリの
        # engine_key をキーとした辞書で持つことで、認証/モデル一覧取得/
        # モデル変更/クライアント更新の4メソッドをエンジンの数だけ書かずに
        # 済む (authenticationRegistryAuthKey 等、下記参照)。
        self._provider_clients: Dict[str, Any] = {}
        self.openai_compatible_client: Optional[Any] = None
        self.lmstudio_client: Optional[Any] = None
        self.lmstudio_connected: bool = False
        self.ollama_client: Optional[Any] = None
        self.ollama_connected: bool = False
        self.ctranslate2_translator: Any = None
        self.ctranslate2_tokenizer: Any = None
        self.is_loaded_ctranslate2_model: bool = False
        self.is_changed_translator_parameters: bool = False
        # ctranslate2_translator/ctranslate2_tokenizer は mic/speaker の
        # _print_transcript スレッドと mainloop ワーカー (チャット送信) から
        # 同時に触られる共有状態。tokenizer.src_lang への代入は tokenizer
        # インスタンス自体の状態を書き換えるため、無ロックだと mic/speaker
        # 間で src_lang が競合して意図しない言語で encode/decode されたり
        # (クラッシュせず気付きにくい翻訳品質の劣化)、changeCTranslate2Model()
        # によるモデル差し替え中の破棄途中オブジェクトに translate_batch が
        # 入りうる。CTranslate2 の Translator 自体はスレッドセーフだが、
        # これは tokenizer の保護にはならない。RLock なのは、将来
        # 呼び出し元が既にロックを保持した状態で再入する可能性に備えるため
        # (現状のコードパスでは再入は起きない)。
        self._ctranslate2_lock: RLock = RLock()
        self._cache = _TranslationCache()

    def authenticationDeepLAuthKey(self, auth_key: str) -> bool:
        """Authenticate DeepL API with the provided key.

        Returns True on success, False on failure.
        """
        result = True
        try:
            self.deepl_client = DeepLClient(auth_key)
            # quick smoke test
            self.deepl_client.translate_text(" ", target_lang="EN-US")
        except Exception:
            errorLogging()
            self.deepl_client = None
            result = False
        return result

    def authenticationRegistryAuthKey(self, engine_key: str, auth_key: str, root_path: str = None, **client_kwargs) -> bool:
        """`TRANSLATION_PROVIDER_REGISTRY` 登録エンジン共通の認証処理。

        `client_kwargs` は OpenAI_API の `base_url` のように、一部エンジンの
        クライアントコンストラクタだけが受け取る追加引数を通すためのもの
        (Plamo/Gemini/Groq/OpenRouter は指定不要)。

        Returns True on success, False on failure.
        """
        client = TRANSLATION_PROVIDER_REGISTRY[engine_key].client_class(root_path=root_path, **client_kwargs)
        if client.setAuthKey(auth_key):
            self._provider_clients[engine_key] = client
            return True
        else:
            self._provider_clients.pop(engine_key, None)
            return False

    def getRegistryModelList(self, engine_key: str) -> list[str]:
        """`TRANSLATION_PROVIDER_REGISTRY` 登録エンジン共通のモデル一覧取得。

        Returns a list of model names, or an empty list on failure.
        """
        client = self._provider_clients.get(engine_key)
        if client is None:
            return []
        return client.getModelList()

    def setRegistryModel(self, engine_key: str, model: str) -> bool:
        """`TRANSLATION_PROVIDER_REGISTRY` 登録エンジン共通のモデル変更。

        Returns True on success, False on failure.
        """
        client = self._provider_clients.get(engine_key)
        if client is None:
            return False
        return client.setModel(model)

    def updateRegistryClient(self, engine_key: str) -> None:
        """`TRANSLATION_PROVIDER_REGISTRY` 登録エンジン共通のクライアント更新。"""
        client = self._provider_clients.get(engine_key)
        if client is not None:
            client.updateClient()

    def authenticationPlamoAuthKey(self, auth_key: str, root_path: str = None) -> bool:
        """Authenticate Plamo API with the provided key.

        Returns True on success, False on failure.
        """
        return self.authenticationRegistryAuthKey("Plamo_API", auth_key, root_path=root_path)

    def getPlamoModelList(self) -> list[str]:
        """Get available Plamo models.

        Returns a list of model names, or an empty list on failure.
        """
        return self.getRegistryModelList("Plamo_API")

    def setPlamoModel(self, model: str) -> bool:
        """Change the Plamo model used for translation.

        Returns True on success, False on failure.
        """
        return self.setRegistryModel("Plamo_API", model)

    def updatePlamoClient(self) -> None:
        """Update the Plamo client (fetch available models)."""
        self.updateRegistryClient("Plamo_API")

    def authenticationGeminiAuthKey(self, auth_key: str, root_path: str = None) -> bool:
        """Authenticate Gemini API with the provided key.

        Returns True on success, False on failure.
        """
        return self.authenticationRegistryAuthKey("Gemini_API", auth_key, root_path=root_path)

    def getGeminiModelList(self) -> list[str]:
        """Get available Gemini models.

        Returns a list of model names, or an empty list on failure.
        """
        return self.getRegistryModelList("Gemini_API")

    def setGeminiModel(self, model: str) -> bool:
        """Change the Gemini model used for translation.

        Returns True on success, False on failure.
        """
        return self.setRegistryModel("Gemini_API", model)

    def updateGeminiClient(self) -> None:
        """Update the Gemini client (fetch available models)."""
        self.updateRegistryClient("Gemini_API")

    def authenticationOpenAIAuthKey(self, auth_key: str, base_url: str | None = None, root_path: str = None) -> bool:
        """Authenticate OpenAI (Chat Completions) API with the provided key.

        base_url を指定することで互換エンドポイント (例: Azure OpenAI 互換, Proxy) にも対応可能。
        Returns True on success, False on failure.
        """
        return self.authenticationRegistryAuthKey("OpenAI_API", auth_key, root_path=root_path, base_url=base_url)

    def getOpenAIModelList(self) -> list[str]:
        """Get available OpenAI models.

        Returns a list of model names, or an empty list on failure.
        """
        return self.getRegistryModelList("OpenAI_API")

    def setOpenAIModel(self, model: str) -> bool:
        """Change the OpenAI model used for translation.

        Returns True on success, False on failure.
        """
        return self.setRegistryModel("OpenAI_API", model)

    def updateOpenAIClient(self) -> None:
        """Update the OpenAI client (fetch available models)."""
        self.updateRegistryClient("OpenAI_API")

    def authenticationOpenAICompatibleAuthKey(self, auth_key: str, base_url: str | None = None, root_path: str = None) -> bool:
        """Authenticate an OpenAI-compatible endpoint with the provided key and base URL.

        `base_url` は必須想定（None の場合は公式エンドポイントにフォールバック）。
        Returns True on success, False on failure.
        """
        self.openai_compatible_client = OpenAICompatibleClient(base_url=base_url, root_path=root_path)
        if self.openai_compatible_client.setAuthKey(auth_key):
            return True
        else:
            self.openai_compatible_client = None
            return False

    def getOpenAICompatibleModelList(self) -> list[str]:
        """Get available OpenAI-compatible endpoint models."""
        if self.openai_compatible_client is None:
            return []
        return self.openai_compatible_client.getModelList()

    def setOpenAICompatibleModel(self, model: str) -> bool:
        """Change the OpenAI-compatible model used for translation."""
        if self.openai_compatible_client is None:
            return False
        return self.openai_compatible_client.setModel(model)

    def updateOpenAICompatibleClient(self) -> None:
        """Update the OpenAI-compatible client (fetch available models)."""
        self.openai_compatible_client.updateClient()

    def authenticationGroqAuthKey(self, auth_key: str, root_path: str = None) -> bool:
        """Authenticate Groq API with the provided key.

        Returns True on success, False on failure.
        """
        return self.authenticationRegistryAuthKey("Groq_API", auth_key, root_path=root_path)

    def getGroqModelList(self) -> list[str]:
        """Get available Groq models.

        Returns a list of model names, or an empty list on failure.
        """
        return self.getRegistryModelList("Groq_API")

    def setGroqModel(self, model: str) -> bool:
        """Change the Groq model used for translation.

        Returns True on success, False on failure.
        """
        return self.setRegistryModel("Groq_API", model)

    def updateGroqClient(self) -> None:
        """Update the Groq client (fetch available models)."""
        self.updateRegistryClient("Groq_API")

    def authenticationOpenRouterAuthKey(self, auth_key: str, root_path: str = None) -> bool:
        """Authenticate OpenRouter API with the provided key.

        Returns True on success, False on failure.
        """
        return self.authenticationRegistryAuthKey("OpenRouter_API", auth_key, root_path=root_path)

    def getOpenRouterModelList(self) -> list[str]:
        """Get available OpenRouter models.

        Returns a list of model names, or an empty list on failure.
        """
        return self.getRegistryModelList("OpenRouter_API")

    def setOpenRouterModel(self, model: str) -> bool:
        """Change the OpenRouter model used for translation.

        Returns True on success, False on failure.
        """
        return self.setRegistryModel("OpenRouter_API", model)

    def updateOpenRouterClient(self) -> None:
        """Update the OpenRouter client (fetch available models)."""
        self.updateRegistryClient("OpenRouter_API")

    def getLMStudioConnected(self) -> bool:
        """Get LM Studio connection status.

        Returns True if connected and verified, False otherwise.
        """
        return self.lmstudio_connected

    def setLMStudioClientURL(self, base_url: str | None = None, root_path: str = None) -> bool:
        """Authenticate LM Studio with the provided base URL.

        Returns True on success, False on failure.
        """
        self.lmstudio_client = LMStudioClient(base_url=base_url, root_path=root_path)
        result = self.lmstudio_client.setBaseURL(base_url)
        if result is False:
            self.lmstudio_client = None
            self.lmstudio_connected = False
        else:
            self.lmstudio_connected = True
        return result

    def getLMStudioModelList(self) -> list[str]:
        """Get available LM Studio models.

        Returns a list of model names, or an empty list on failure.
        """
        if self.lmstudio_client is None:
            return []
        return self.lmstudio_client.getModelList()

    def setLMStudioModel(self, model: str) -> bool:
        """Change the LM Studio model used for translation.
        """
        if self.lmstudio_client is None:
            return False
        return self.lmstudio_client.setModel(model)

    def updateLMStudioClient(self) -> None:
        """Update the LM Studio client (fetch available models)."""
        self.lmstudio_client.updateClient()

    def getOllamaConnected(self) -> bool:
        """Get Ollama connection status.

        Returns True if connected and verified, False otherwise.
        """
        return self.ollama_connected

    def checkOllamaClient(self, root_path: str = None) -> bool:
        """Check if Ollama client is available.

        Returns True if Ollama is reachable, False otherwise.
        """
        self.ollama_client = OllamaClient(root_path=root_path)
        result = self.ollama_client.authenticationCheck()
        if result is False:
            self.ollama_client = None
            self.ollama_connected = False
        else:
            self.ollama_connected = True
        return result

    def getOllamaModelList(self, root_path: str = None) -> bool:
        """Initialize Ollama client and fetch available models.

        Returns True on success, False on failure.
        """
        if self.ollama_client is None:
            return []
        return self.ollama_client.getModelList()

    def setOllamaModel(self, model: str) -> bool:
        """Change the Ollama model used for translation.

        Returns True on success, False on failure.
        """
        if self.ollama_client is None:
            return False
        return self.ollama_client.setModel(model)

    def updateOllamaClient(self) -> None:
        """Update the Ollama client (fetch available models)."""
        self.ollama_client.updateClient()

    def changeCTranslate2Model(self, path: str, model_type: str, device: str = "cpu", device_index: int = 0, compute_type: str = "auto") -> None:
        """Load a CTranslate2 model from weights.

        This sets internal translator/tokenizer objects and flips
        ``is_loaded_ctranslate2_model`` on success.
        """
        if ctranslate2 is None or transformers is None:
            return

        with self._ctranslate2_lock:
            self.is_loaded_ctranslate2_model = False
            directory_name = ctranslate2_weights[model_type]["directory_name"]
            tokenizer = ctranslate2_weights[model_type]["tokenizer"]
            weight_path = os_path.join(path, "weights", "ctranslate2", directory_name)
            tokenizer_path = os_path.join(path, "weights", "ctranslate2", directory_name, "tokenizer")

            if compute_type == "auto":
                compute_type = getBestComputeType(device, device_index)
            self.ctranslate2_translator = ctranslate2.Translator(
                weight_path,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                inter_threads=1,
                intra_threads=4,
            )
            try:
                self.ctranslate2_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer, cache_dir=tokenizer_path)
            except Exception:
                errorLogging()
                tokenizer_path = os_path.join("./weights", "ctranslate2", directory_name, "tokenizer")
                self.ctranslate2_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer, cache_dir=tokenizer_path)
            self.is_loaded_ctranslate2_model = True
            # 重みが変われば訳も変わる (キーに weight_type は含むが、同じ重みの
            # 再ロードでも念のため捨てる)
            self._cache.clear()

    def isLoadedCTranslate2Model(self) -> bool:
        return self.is_loaded_ctranslate2_model

    def isChangedTranslatorParameters(self) -> bool:
        return self.is_changed_translator_parameters

    def setChangedTranslatorParameters(self, is_changed: bool) -> None:
        self.is_changed_translator_parameters = is_changed

    def _ctranslate2TargetToken(self, target_language: str, weight_type: str) -> Optional[str]:
        match weight_type:
            case "m2m100_418M-ct2-int8" | "m2m100_1.2B-ct2-int8":
                return self.ctranslate2_tokenizer.lang_code_to_token[target_language]
            case "nllb-200-distilled-600M-ct2-int8" | "nllb-200-distilled-1.3B-ct2-int8" | "nllb-200-3.3B-ct2-int8":
                return target_language
            case _:
                return None

    def translateCTranslate2Batch(
        self,
        message: str,
        source_language: str,
        target_languages: List[str],
        weight_type: str,
        source_name: Optional[str] = None,
        target_names: Optional[List[str]] = None,
    ) -> List[Any]:
        """Translate `message` into every language code in `target_languages`
        with a single ``translate_batch`` call.

        The message is split into sentences (NLLB/M2M100 are sentence-level
        models and drop or garble content when given several sentences at
        once), and every (sentence, target) pair goes into one batch. With
        several target languages this replaces N sequential decodes - the
        lock serialises them anyway - with one batched decode.

        `source_name` / `target_names` are the display names of the languages
        ("Japanese", "English"). When given, chat laughter ("www", "草",
        "ㅋㅋㅋ", "lol") is taken out before translating and written back the
        way the target language writes it: the models invent sentences for it
        otherwise (translation_laughter.py).

        Returns one entry per target: a string on success, False on failure.
        """
        results: List[Any] = [False] * len(target_languages)
        if not target_languages:
            return results

        # (is_laughter, text) in message order. Without language names the
        # whole message is one text piece, as before.
        pieces = [(False, message)]
        if source_name is not None and target_names is not None and len(target_names) == len(target_languages):
            pieces = splitLaughter(message, source_name)
        text_pieces = [text for is_laughter, text in pieces if not is_laughter]
        if not text_pieces:
            # Nothing but laughter: no need for the model at all.
            return [joinLaughter(pieces, [], name) for name in target_names]

        with self._ctranslate2_lock:
            if self.is_loaded_ctranslate2_model is not True:
                return results
            try:
                prefixes = [self._ctranslate2TargetToken(t, weight_type) for t in target_languages]
                if any(p is None for p in prefixes):
                    return results
                tokenizer = self.ctranslate2_tokenizer
                tokenizer.src_lang = source_language
                # Sentences of each text piece, tokenised. Short sentences are
                # only merged with neighbours of the same piece.
                piece_sources: List[List[List[str]]] = []
                for text in text_pieces:
                    sentences = splitSentences(text)
                    sources = [tokenizer.convert_ids_to_tokens(tokenizer.encode(s)) for s in sentences]
                    if len(sentences) > 1:
                        merged = mergeShortSentences(sentences, [len(s) for s in sources])
                        if len(merged) != len(sentences):
                            sentences = merged
                            sources = [tokenizer.convert_ids_to_tokens(tokenizer.encode(s)) for s in sentences]
                    piece_sources.append(sources)
                sources = [source for piece in piece_sources for source in piece]

                batch_sources = []
                batch_prefixes = []
                for prefix in prefixes:
                    for source in sources:
                        batch_sources.append(source)
                        batch_prefixes.append([prefix])
                max_decoding_length = min(
                    _CT2_DECODING_LENGTH_MAX,
                    max(len(s) for s in sources) * _CT2_DECODING_LENGTH_RATIO + _CT2_DECODING_LENGTH_MARGIN,
                )
                outputs = self.ctranslate2_translator.translate_batch(
                    batch_sources,
                    target_prefix=batch_prefixes,
                    max_decoding_length=max_decoding_length,
                )

                n = len(sources)
                for i in range(len(prefixes)):
                    decoded = []
                    for j in range(n):
                        target = outputs[i * n + j].hypotheses[0][1:]
                        decoded.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
                    translated_pieces = []
                    offset = 0
                    for piece in piece_sources:
                        piece_decoded = decoded[offset:offset + len(piece)]
                        offset += len(piece)
                        translated_pieces.append(
                            piece_decoded[0] if len(piece_decoded) == 1 else joinSentences(piece_decoded)
                        )
                    if len(pieces) == 1:
                        results[i] = translated_pieces[0]
                    else:
                        results[i] = joinLaughter(pieces, translated_pieces, target_names[i])
            except Exception:
                errorLogging()
                results = [False] * len(target_languages)
        return results

    def translateCTranslate2(
        self,
        message: str,
        source_language: str,
        target_language,
        weight_type: str,
        source_name: Optional[str] = None,
        target_name: Optional[str] = None,
    ) -> Any:
        """Translate using a loaded CTranslate2 model.

        Returns a string on success or False on failure (keeps legacy behavior).
        """
        return self.translateCTranslate2Batch(
            message,
            source_language,
            [target_language],
            weight_type,
            source_name=source_name,
            target_names=None if target_name is None else [target_name],
        )[0]

    @staticmethod
    def _cacheKey(translator_name: str, weight_type: str, source_language: str, target_language: str, message: str) -> Tuple:
        # target_country は DeepL の英語/ポルトガル語の変種選択にしか使われず、
        # DeepL はキャッシュ対象外なのでキーに含めない (国違いでも同じ訳を共有する)。
        return (
            translator_name,
            weight_type if translator_name == "CTranslate2" else "",
            source_language,
            target_language,
            message,
        )

    def translateCTranslate2Targets(self, weight_type: str, source_language: str, targets: List[Tuple[str, str]], message: str) -> List[Any]:
        """Translate `message` with CTranslate2 into several target languages at once.

        `targets` is a list of (language name, country). Each entry of the
        result follows the `translate()` contract: str on success, None if the
        language pair is unsupported, False on failure.
        """
        results: List[Any] = [None] * len(targets)
        pending = []  # (index, cache key, target code)
        source_code = None
        for i, (target_language, target_country) in enumerate(targets):
            if source_language == target_language:
                results[i] = message
                continue
            try:
                src_code, tgt_code = self.getLanguageCode("CTranslate2", weight_type, target_country, source_language, target_language)
            except UnsupportedLanguageError:
                results[i] = None
                continue
            key = self._cacheKey("CTranslate2", weight_type, source_language, target_language, message)
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
                continue
            source_code = src_code
            pending.append((i, key, tgt_code))

        if pending:
            translated = self.translateCTranslate2Batch(
                message,
                source_code,
                [p[2] for p in pending],
                weight_type,
                source_name=source_language,
                target_names=[targets[p[0]][0] for p in pending],
            )
            for (i, key, _), result in zip(pending, translated):
                if isinstance(result, str) and (result.strip() or not message.strip()):
                    self._cache.put(key, result)
                    results[i] = result
                else:
                    results[i] = False
        return results

    @staticmethod
    def _deeplContext(context_history: Optional[list[dict]]) -> Optional[str]:
        """DeepL の `context` に渡す直近の会話。訳の曖昧さ (代名詞・省略された
        主語・口調) の解消に使われ、訳文には含まれない。課金対象外。"""
        if not context_history:
            return None
        texts = [h.get("text", "").strip() for h in context_history[-_DEEPL_CONTEXT_MAX_MESSAGES:]]
        context = "\n".join(t for t in texts if t)
        if len(context) > _DEEPL_CONTEXT_MAX_CHARS:
            context = context[-_DEEPL_CONTEXT_MAX_CHARS:]
        return context or None

    @staticmethod
    def getLanguageCode(translator_name: str, weight_type: str, target_country: str, source_language: str, target_language: str) -> Tuple[str, str]:
        """Resolve a friendly language name to translator-specific codes.

        Returns (source_code, target_code).

        Raises:
            UnsupportedLanguageError: `source_language`/`target_language` is
                not in this translator's supported language table.
        """
        try:
            match translator_name:
                case "DeepL_API":
                    if target_language == "English":
                        if target_country in ["United States", "Canada", "Philippines"]:
                            target_language = "English American"
                        else:
                            target_language = "English British"
                    elif target_language == "Portuguese":
                        if target_country in ["Portugal"]:
                            target_language = "Portuguese European"
                        else:
                            target_language = "Portuguese Brazilian"
                    source_language = translation_lang[translator_name]["source"][source_language]
                    target_language = translation_lang[translator_name]["target"][target_language]
                case "CTranslate2":
                    source_language = translation_lang[translator_name][weight_type]["source"][source_language]
                    target_language = translation_lang[translator_name][weight_type]["target"][target_language]
                case _:
                    source_language = translation_lang[translator_name]["source"][source_language]
                    target_language = translation_lang[translator_name]["target"][target_language]
        except KeyError as e:
            raise UnsupportedLanguageError(
                f"{translator_name} does not support language {e} (source={source_language!r}, target={target_language!r})"
            ) from e
        return source_language, target_language

    def translate(self, translator_name: str, weight_type: str, source_language: str, target_language: str, target_country: str, message: str, context_history: Optional[list[dict]] = None) -> Any:
        """Translate `message` using the named translator backend.

        Args:
            translator_name: Name of the translator backend to use
            weight_type: Model weight type for CTranslate2
            source_language: Source language name
            target_language: Target language name
            target_country: Target country for locale-specific translations
            message: Text to translate
            context_history: Optional conversation context (Chat/Mic/Speaker messages)

        Returns translated string on success. Returns None if the language
        pair is not supported by this translator (not a backend failure).
        Returns False on an actual backend failure (rate limit, network,
        authentication, provider error). When source_language ==
        target_language the original message is returned.
        """
        try:
            if source_language == target_language:
                return message

            # False, not "": every `case` below is guarded (engine disabled,
            # client not authenticated, library missing), and an unknown
            # translator_name matches no case at all. Any of those leaves this
            # initial value untouched, and the caller decides success with
            # `isinstance(translation, str)` - so "" would report an empty
            # translation as a success, skipping the CTranslate2 fallback and
            # the error log. False routes it through the fallback instead.
            result: Any = False
            cache_key = None
            if translator_name in _CACHEABLE_ENGINES:
                cache_key = self._cacheKey(translator_name, weight_type, source_language, target_language, message)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return cached
            source_name, target_name = source_language, target_language
            source_language, target_language = self.getLanguageCode(translator_name, weight_type, target_country, source_language, target_language)
            match translator_name:
                case "DeepL_API":
                    if self.is_enable_translators is True:
                        if self.deepl_client is None:
                            result = False
                        else:
                            deepl_kwargs = {}
                            context = self._deeplContext(context_history)
                            if context:
                                deepl_kwargs["context"] = context
                            result = self.deepl_client.translate_text(
                                message,
                                source_lang=source_language,
                                target_lang=target_language,
                                **deepl_kwargs,
                                ).text
                case name if name in TRANSLATION_PROVIDER_REGISTRY:
                    # Plamo_API/Gemini_API/OpenAI_API/Groq_API/OpenRouter_API
                    # (「認証キー+モデル一覧」型、フェーズ3項目17) は
                    # クライアントのインターフェースが完全に同一なため、
                    # 1つの分岐にまとめられる。OpenAI_Compatible は
                    # base_url依存・別クラス継承のためレジストリ対象外
                    # (下の専用 case のまま)。
                    provider_client = self._provider_clients.get(name)
                    if provider_client is None:
                        result = False
                    else:
                        if context_history:
                            provider_client.setContextHistory(context_history)
                        result = provider_client.translate(
                            message,
                            input_lang=source_language,
                            output_lang=target_language,
                            )
                case "OpenAI_Compatible":
                    if self.openai_compatible_client is None:
                        result = False
                    else:
                        if context_history:
                            self.openai_compatible_client.setContextHistory(context_history)
                        result = self.openai_compatible_client.translate(
                            message,
                            input_lang=source_language,
                            output_lang=target_language,
                        )
                case "LMStudio":
                    if self.lmstudio_client is None:
                        result = False
                    else:
                        if context_history:
                            self.lmstudio_client.setContextHistory(context_history)
                        result = self.lmstudio_client.translate(
                            message,
                            input_lang=source_language,
                            output_lang=target_language,
                        )
                case "Ollama":
                    if self.ollama_client is None:
                        result = False
                    else:
                        if context_history:
                            self.ollama_client.setContextHistory(context_history)
                        result = self.ollama_client.translate(
                            message,
                            input_lang=source_language,
                            output_lang=target_language,
                        )
                case "Google":
                    if ENABLE_TRANSLATORS is True and other_web_Translator is not None:
                        result = other_web_Translator(
                            query_text=message,
                            translator="google",
                            from_language=source_language,
                            to_language=target_language,
                            timeout=_WEB_TRANSLATOR_TIMEOUT_SECONDS,
                        )
                case "Bing":
                    if ENABLE_TRANSLATORS is True and other_web_Translator is not None:
                        result = other_web_Translator(
                            query_text=message,
                            translator="bing",
                            from_language=source_language,
                            to_language=target_language,
                            timeout=_WEB_TRANSLATOR_TIMEOUT_SECONDS,
                        )
                case "Papago":
                    if ENABLE_TRANSLATORS is True and other_web_Translator is not None:
                        result = other_web_Translator(
                            query_text=message,
                            translator="papago",
                            from_language=source_language,
                            to_language=target_language,
                            timeout=_WEB_TRANSLATOR_TIMEOUT_SECONDS,
                        )
                case "CTranslate2":
                    result = self.translateCTranslate2(
                        message=message,
                        source_language=source_language,
                        target_language=target_language,
                        weight_type=weight_type,
                        source_name=source_name,
                        target_name=target_name,
                        )
            if isinstance(result, str):
                if not result.strip() and message.strip():
                    # 空の訳は成功ではない。呼び出し元は isinstance(str) で成功を
                    # 判定するので、"" を返すと CTranslate2 へのフォールバックが
                    # 飛ばされ、空のメッセージが送られてしまう。
                    result = False
                elif cache_key is not None:
                    self._cache.put(cache_key, result)
        except UnsupportedLanguageError:
            result = None
        except Exception:
            errorLogging()
            result = False
        return result

if __name__ == "__main__":
    translator = Translator()
    # test CTranslate2 model nllb-200-distilled-1.3B-ct2-int8
    translator.changeCTranslate2Model(path=".", model_type="nllb-200-distilled-1.3B-ct2-int8", device="cpu", device_index=0)
    result = translator.translate(
        translator_name="CTranslate2",
        weight_type="nllb-200-distilled-1.3B-ct2-int8",
        source_language="English",
        target_language="Japanese",
        target_country="Japan",
        message="Hello, world!"
        )
    print(result)