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

BASE_URL = "https://api.platform.preferredai.jp/v1"

def _fetch_models(api_key: str):
    """モデル一覧を取得する。成功すればキーは有効 (認証の確認を兼ねる)。失敗は例外。"""
    return OpenAI(api_key=api_key, base_url=BASE_URL).models.list()

def _text_models(res) -> list[str]:
    """Extract all available models from the PLAMO API
    """
    allowed_models = []

    for model in res.data:
        allowed_models.append(model.id)

    allowed_models.sort()
    return allowed_models

def _get_available_text_models(api_key: str) -> list[str]:
    return _text_models(_fetch_models(api_key))

class PlamoClient(LLMTranslationMixin, RecentModelList):
    def __init__(self, root_path: str = None):
        self.api_key = None
        self.base_url = BASE_URL
        self.model = None

        self._initPrompt(root_path, "translation_plamo.yml", "Plamo_API")

        self.plamo_llm = None

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
        self.plamo_llm = ChatOpenAI(
            base_url=self.base_url,
            model=self.model,
            streaming=False,
            api_key=SecretStr(self.api_key),
        )

    def translate(self, text: str, input_lang: str, output_lang: str) -> str:
        return self._translateWith(self.plamo_llm, text, input_lang, output_lang)

if __name__ == "__main__":
    AUTH_KEY = "PLAMO_API_KEY"
    client = PlamoClient()
    client.setAuthKey(AUTH_KEY)
    models = client.getModelList()
    if models:
        print("Available models:", models)
        model = input("Select a model: ")
        client.setModel(model)
        client.updateClient()
        print(client.translate("こんにちは世界", "Japanese", "English"))