import logging
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from .translation_model_list import RecentModelList
    from .translation_llm_base import LLMTranslationMixin
except Exception:
    import sys
    from os import path as os_path
    print(os_path.dirname(os_path.dirname(os_path.dirname(os_path.abspath(__file__)))))
    sys.path.append(os_path.dirname(os_path.dirname(os_path.dirname(os_path.abspath(__file__)))))
    from translation_model_list import RecentModelList
    from translation_llm_base import LLMTranslationMixin

logger = logging.getLogger("langchain_google_genai")
logger.setLevel(logging.ERROR)

class _ModelListing:
    """models.list() の結果と、それを取った Client をひとまとめにしたもの。

    Client は破棄されると内部の HTTP クライアントを閉じる。`.models` は Client 自身ではなく
    HTTP クライアントだけを持つので、Client を手元に持たないと、`list()` の最中や、次のページを
    取るとき ("Client has been closed") に失敗する。以前は `genai.Client(...).models.list()` と
    つないでいたため、有効なキーでも認証が失敗し、キーが無効扱いになって設定から消えた。
    """

    def __init__(self, client, pager):
        self.client = client
        self.pager = pager

    def __iter__(self):
        return iter(self.pager)


def _fetch_models(api_key: str):
    """モデル一覧の最初のページを取得する。成功すればキーは有効 (認証の確認を兼ねる)。失敗は例外。"""
    client = genai.Client(api_key=api_key)
    return _ModelListing(client, client.models.list())

def _text_models(res) -> list[str]:
    """Extract only Gemini models suitable for translation and chat applications

    res は全ページを順にたどれる (たどる間に次のページを取得する)。
    """
    allowed_models = []

    # 除外対象のキーワード
    exclude_keywords = [
        "audio",
        "image",
        "veo",
        "tts",
        "robotics",
        "computer-use"
    ]
    for model in res:
        model_id = model.name
        if ("gemini" in model_id.lower() or "gemma" in model_id.lower()) and "generateContent" in model.supported_actions:
            if any(x in model_id for x in exclude_keywords):
                continue
            allowed_models.append(model_id.replace("models/", ""))
    allowed_models.sort()
    return allowed_models

def _get_available_text_models(api_key: str) -> list[str]:
    return _text_models(_fetch_models(api_key))

class GeminiClient(LLMTranslationMixin, RecentModelList):
    def __init__(self, root_path: str = None):
        self.api_key = None
        self.model = None

        # プロンプト設定をYAMLファイルから読み込む
        self._initPrompt(root_path, "translation_gemini.yml", "Gemini_API")

        self.gemini_llm = None

    def getModelList(self) -> list[str]:
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
        self.gemini_llm = ChatGoogleGenerativeAI(
            model=self.model,
            api_key=self.api_key,
        )

    def translate(self, text: str, input_lang: str, output_lang: str) -> str:
        return self._translateWith(self.gemini_llm, text, input_lang, output_lang)

if __name__ == "__main__":
    AUTH_KEY = "AUTH_KEY"
    client = GeminiClient()
    client.setAuthKey(AUTH_KEY)
    models = client.getModelList()
    if models:
        print("Available models:", models)
        model = input("Select a model: ")
        client.setModel(model)
        client.updateClient()
        print(client.translate("こんにちは世界", "Japanese", "English"))