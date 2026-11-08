from openai import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

try:
    from .translation_languages import translation_lang
    from .translation_model_list import RecentModelList
    from .translation_llm_base import LLMTranslationMixin
except Exception:
    import sys
    from os import path as os_path
    sys.path.append(os_path.dirname(os_path.dirname(os_path.dirname(os_path.abspath(__file__)))))
    from translation_languages import translation_lang, loadTranslationLanguages
    from translation_model_list import RecentModelList
    from translation_llm_base import LLMTranslationMixin
    translation_lang = loadTranslationLanguages(path=".", force=True)

def _fetch_models(api_key: str, base_url: str | None = None):
    """モデル一覧を取得する。成功すればキーは有効 (認証の確認を兼ねる)。失敗は例外。"""
    return OpenAI(api_key=api_key, base_url=base_url).models.list()

def _text_models(res) -> list[str]:
    """Extract only GPT models suitable for translation and chat applications (plus those with fine-tuning)
    """
    allowed_models = []

    for model in res.data:
        model_id = model.id
        root = getattr(model, "root", "")

        # 除外対象のキーワード
        exclude_keywords = [
            "whisper",       # 音声認識
            "embedding",     # 埋め込み
            "image",         # 画像生成
            "tts",           # 音声合成
            "audio",         # 音声系（transcribe, diarize含む）
            "search",        # 検索補助モデル
            "transcribe",    # 音声→文字起こし
            "diarize",       # 話者分離
            "vision"         # 画像入力系（旧gpt-4-visionなど）
        ]

        # 除外キーワードが含まれているモデルをスキップ
        if any(kw in model_id for kw in exclude_keywords):
            continue

        # GPTモデルまたはFine-tune GPTモデルのみ対象
        if model_id.startswith("gpt-"):
            allowed_models.append(model_id)
        elif model_id.startswith("ft:") and root.startswith("gpt-"):
            allowed_models.append(model_id)

    allowed_models.sort()
    return allowed_models

def _get_available_text_models(api_key: str, base_url: str | None = None) -> list[str]:
    return _text_models(_fetch_models(api_key, base_url))

class OpenAIClient(LLMTranslationMixin, RecentModelList):
    """OpenAI Translation simple wrapper.
    prompt/translation_openai.yml から system_prompt / supported_languages を読み込む。
    """
    def __init__(self, base_url: str | None = None, root_path: str = None):
        self.api_key = None
        self.model = None
        self.base_url = base_url  # None の場合は公式エンドポイント

        self._initPrompt(root_path, "translation_openai.yml", "OpenAI_API")

        self.openai_llm = None

    def getModelList(self) -> list[str]:
        if not self.api_key:
            return []
        return self._remember(self._takeAuthModels() or _get_available_text_models(self.api_key, self.base_url))

    def getAuthKey(self) -> str:
        return self.api_key

    def setAuthKey(self, api_key: str) -> bool:
        try:
            res = _fetch_models(api_key, self.base_url)
        except Exception:
            return False
        self.api_key = api_key
        self._rememberFromAuth(lambda: _text_models(res))
        return True

    def getModel(self) -> str:
        return self.model

    def updateClient(self) -> None:
        self.openai_llm = ChatOpenAI(
            base_url=self.base_url,
            model=self.model,
            api_key=SecretStr(self.api_key),
            streaming=False,
        )

    def translate(self, text: str, input_lang: str, output_lang: str) -> str:
        return self._translateWith(self.openai_llm, text, input_lang, output_lang)

if __name__ == "__main__":
    AUTH_KEY = input("OPENAI_API_KEY: ")
    client = OpenAIClient()
    client.setAuthKey(AUTH_KEY)
    models = client.getModelList()
    if models:
        print("Available models:", models)
        model = input("Select a model: ")
        client.setModel(model)
        client.updateClient()
        print(client.translate("こんにちは世界", "Japanese", "English"))