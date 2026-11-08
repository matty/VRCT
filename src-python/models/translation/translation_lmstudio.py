from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import requests

try:
    from .translation_model_list import RecentModelList
    from .translation_llm_base import LLMTranslationMixin
except Exception:
    import sys
    from os import path as os_path
    sys.path.append(os_path.dirname(os_path.abspath(__file__)))
    from translation_languages import loadTranslationLanguages
    from translation_model_list import RecentModelList
    from translation_llm_base import LLMTranslationMixin
    loadTranslationLanguages(path=".", force=True)

def _authentication_check(base_url: str | None = None) -> bool:
    """Check if the provided API key is valid by attempting to list models.
    """
    try:
        response = requests.get(f"{base_url}/models", timeout=0.2)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception:
        return False

def _get_available_text_models(base_url: str | None = None) -> list[str]:
    """Extract the list of available text models from the LM Studio.
    """
    try:
        response = requests.get(f"{base_url}/models", timeout=0.2)
        models = response.json()["data"]
    except Exception:
        models = []

    allowed_models = []
    for model in models:
        allowed_models.append(model["id"])

    allowed_models.sort()
    return allowed_models

class LMStudioClient(LLMTranslationMixin, RecentModelList):
    """LM Studio Translation simple wrapper.
    prompt/translation_lmstudio.yml から system_prompt / supported_languages を読み込む。
    """
    def __init__(self, base_url: str | None = None, root_path: str = None):
        self.api_key = "lmstudio"
        self.model = None
        self.base_url = base_url  # None の場合は公式エンドポイント

        self._initPrompt(root_path, "translation_lmstudio.yml", "LMStudio")

        self.openai_llm = None

    def getBaseURL(self) -> str | None:
        return self.base_url

    def setBaseURL(self, base_url: str | None) -> None:
        result = _authentication_check(base_url=base_url)
        if result:
            self.base_url = base_url
        return result

    def getModelList(self) -> list[str]:
        return self._remember(_get_available_text_models(base_url=self.base_url)) if self.base_url else []

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
    client = LMStudioClient(base_url="http://127.0.0.1:1234/v1")
    models = client.getModelList()
    if models:
        print("Available models:", models)
        model = input("Select a model: ")
        client.setModel(model)
        client.updateClient()
        print(client.translate("こんにちは世界", "Japanese", "English"))