from openai import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

try:
    from .translation_model_list import RecentModelList
    from .translation_llm_base import LLMTranslationMixin
except Exception:
    import sys
    from os import path as os_path
    sys.path.append(os_path.dirname(os_path.dirname(os_path.dirname(os_path.abspath(__file__)))))
    from translation_languages import loadTranslationLanguages
    from translation_model_list import RecentModelList
    from translation_llm_base import LLMTranslationMixin
    loadTranslationLanguages(path=".", force=True)

def _fetch_models(api_key: str):
    """モデル一覧を取得する。成功すればキーは有効 (認証の確認を兼ねる)。失敗は例外。"""
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    ).models.list()

def _text_models(res) -> list[str]:
    """Extract only Groq models suitable for translation and chat applications.
    """
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

def _get_available_text_models(api_key: str) -> list[str]:
    return _text_models(_fetch_models(api_key))

class GroqClient(LLMTranslationMixin, RecentModelList):
    """Groq API Translation wrapper using OpenAI-compatible endpoint.
    
    Groq provides a fast LLM inference platform with an OpenAI-compatible API.
    The API endpoint: https://api.groq.com/openai/v1
    """
    def __init__(self, root_path: str = None):
        self.api_key = None
        self.model = None
        self.base_url = "https://api.groq.com/openai/v1"

        self._initPrompt(root_path, "translation_groq.yml", "Groq_API")

        self.groq_llm = None

    def getModelList(self) -> list[str]:
        if not self.api_key:
            return []
        return self._remember(self._takeAuthModels() or _get_available_text_models(self.api_key))

    def getAuthKey(self) -> str:
        return self.api_key

    def setAuthKey(self, api_key: str) -> bool:
        try:
            res = _fetch_models(api_key)
        except Exception:
            return False
        self.api_key = api_key
        self._rememberFromAuth(lambda: _text_models(res))
        return True

    def getModel(self) -> str:
        return self.model

    def updateClient(self) -> None:
        self.groq_llm = ChatOpenAI(
            base_url=self.base_url,
            model=self.model,
            api_key=SecretStr(self.api_key),
            streaming=False,
        )

    def translate(self, text: str, input_lang: str, output_lang: str) -> str:
        return self._translateWith(self.groq_llm, text, input_lang, output_lang)

if __name__ == "__main__":
    AUTH_KEY = "GROQ_API_KEY"
    client = GroqClient()
    client.setAuthKey(AUTH_KEY)
    models = client.getModelList()
    if models:
        print("Available models:", models)
        model = input("Select a model: ")
        client.setModel(model)
        client.updateClient()
        print(client.translate("こんにちは世界", "Japanese", "English"))
