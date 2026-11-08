import requests
from langchain_ollama import ChatOllama

try:
    from .translation_languages import translation_lang
    from .translation_model_list import RecentModelList
    from .translation_llm_base import LLMTranslationMixin
except Exception:
    import sys
    from os import path as os_path
    sys.path.append(os_path.dirname(os_path.abspath(__file__)))
    from translation_languages import translation_lang, loadTranslationLanguages
    from translation_model_list import RecentModelList
    from translation_llm_base import LLMTranslationMixin
    translation_lang = loadTranslationLanguages(path=".", force=True)

def _authentication_check(base_url: str | None = None) -> bool:
    """Check authentication for Ollama API.
    """
    try:
        response = requests.get(f"{base_url}", timeout=0.2)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception:
        return False

def _get_available_text_models(base_url: str | None = None) -> list[str]:
    """Extract available text models from Ollama.
    """
    try:
        response = requests.get(f"{base_url}/api/tags")
        models = response.json()["models"]
    except Exception:
        models = []

    allowed_models = []
    for model in models:
        allowed_models.append(model["name"])

    allowed_models.sort()
    return allowed_models

class OllamaClient(LLMTranslationMixin, RecentModelList):
    """Ollama Translation simple wrapper.
    prompt/translation_ollama.yml から system_prompt / supported_languages を読み込む。
    """
    def __init__(self, root_path: str = None):
        self.model = None
        self.base_url = "http://localhost:11434"

        self._initPrompt(root_path, "translation_ollama.yml", "Ollama")

        self.openai_llm = None

    def authenticationCheck(self) -> bool:
        return _authentication_check(self.base_url)

    def getModelList(self) -> list[str]:
        if self.authenticationCheck():
            return self._remember(_get_available_text_models(self.base_url))
        return []

    def getModel(self) -> str:
        return self.model

    def updateClient(self) -> None:
        self.openai_llm = ChatOllama(
            base_url=self.base_url,
            model=self.model,
            streaming=False,
        )

    def translate(self, text: str, input_lang: str, output_lang: str) -> str:
        return self._translateWith(self.openai_llm, text, input_lang, output_lang)

if __name__ == "__main__":
    client = OllamaClient()
    models = client.getModelList()
    if models:
        print("Available models:", models)
        model = input("Select a model: ")
        client.setModel(model)
        client.updateClient()
        print(client.translate("こんにちは世界", "Japanese", "English"))