import requests
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

def _authentication_check(api_key: str) -> bool:
    """Check if the provided API key is valid by attempting to list models.
    """

    url = "https://openrouter.ai/api/v1/auth/key"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    r = requests.get(url, headers=headers, timeout=10)

    return r.status_code == 200

def _get_available_text_models(api_key: str, base_url: str | None = None) -> list[str]:
    """Extract only OpenRouter models suitable for translation and chat applications.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    res = client.models.list()
    allowed_models = []

    for model in res.data:
        model_id = model.id

        # 除外対象のキーワード
        exclude_keywords = [
            "whisper",       # 音声認識
            "embedding",     # 埋め込み
            "image",         # 画像生成
            "tts",           # 音声合成
            "audio",         # 音声系
            "search",        # 検索補助モデル
            "transcribe",    # 音声→文字起こし
            "diarize",       # 話者分離
            "vision"         # 画像入力系
        ]

        # 除外キーワードが含まれているモデルをスキップ
        if any(kw in model_id.lower() for kw in exclude_keywords):
            continue

        # テキスト処理用モデルのみ対象
        allowed_models.append(model_id)

    allowed_models.sort()
    return allowed_models

class OpenRouterClient(LLMTranslationMixin, RecentModelList):
    """OpenRouter API Translation wrapper using OpenAI-compatible endpoint.
    
    OpenRouter provides access to various LLM models via a unified API.
    The API endpoint: https://openrouter.ai/api/v1
    """
    def __init__(self, root_path: str = None):
        self.api_key = None
        self.model = None
        self.base_url = "https://openrouter.ai/api/v1"

        self._initPrompt(root_path, "translation_openrouter.yml", "OpenRouter_API")

        self.openrouter_llm = None

    def getModelList(self) -> list[str]:
        return self._remember(_get_available_text_models(self.api_key, self.base_url)) if self.api_key else []

    def getAuthKey(self) -> str:
        return self.api_key

    def setAuthKey(self, api_key: str) -> bool:
        result = _authentication_check(api_key)
        if result:
            self.api_key = api_key
        return result

    def getModel(self) -> str:
        return self.model

    def updateClient(self) -> None:
        self.openrouter_llm = ChatOpenAI(
            base_url=self.base_url,
            model=self.model,
            api_key=SecretStr(self.api_key),
            streaming=False,
        )

    def translate(self, text: str, input_lang: str, output_lang: str) -> str:
        return self._translateWith(self.openrouter_llm, text, input_lang, output_lang)

if __name__ == "__main__":
    AUTH_KEY = input("OPENROUTER_API_KEY: ")
    client = OpenRouterClient()
    client.setAuthKey(AUTH_KEY)
    models = client.getModelList()
    if models:
        print("Available models:", models)
        model = input("Select a model: ")
        client.setModel(model)
        client.updateClient()
        print(client.translate("こんにちは世界", "Japanese", "English"))