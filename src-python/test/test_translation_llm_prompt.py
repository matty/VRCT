"""LLM 翻訳のプロンプト組み立てと出力の後処理 (translation_llm_base) のテスト。

- 既定プロンプトに対応言語の一覧 (数百トークン) を入れない
- 会話履歴は「参照専用」と明示し、発話者を Me/Other で渡す
- 翻訳専用のローカルモデル (Hunyuan-MT / HY-MT1.5 / TranslateGemma) には
  モデルカードの公式テンプレートで問い合わせる
- 推論モデルの <think> や全体を囲む引用符を出力から取り除く
"""

import unittest
from types import SimpleNamespace

from models.translation.translation_llm_base import (
    PROFILE_DEFAULT,
    PROFILE_HUNYUAN_MT,
    PROFILE_HY_MT15,
    PROFILE_TRANSLATEGEMMA,
    buildTranslateGemmaPrompt,
    cleanTranslationOutput,
    detectModelProfile,
    extractContent,
    maxOutputTokens,
)
from models.translation.translation_ollama import OllamaClient
from models.translation.translation_lmstudio import LMStudioClient


class ChatOllama:  # 型名で呼び出し方を切り替えるので、名前だけ本物に合わせる
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content=self.reply)


class ChatOpenAI(ChatOllama):
    pass


_HISTORY = [
    {"source": "speaker", "text": "Where are you from?", "timestamp": "2026-11-08T12:00:00"},
    {"source": "mic", "text": "日本から来ました", "timestamp": "2026-11-08T12:00:05"},
    {"source": "ocr", "text": "nice!", "timestamp": "2026-11-08T12:00:09"},
]


class CleanTranslationOutputTests(unittest.TestCase):
    def test_strips_think_block(self):
        self.assertEqual(cleanTranslationOutput("<think>user wants EN</think>\nHello", "こんにちは"), "Hello")

    def test_strips_dangling_close_tag(self):
        # 開始タグをテンプレート側で付けるモデルは閉じタグだけ出力する
        self.assertEqual(cleanTranslationOutput("reasoning...</think>Hello", "こんにちは"), "Hello")

    def test_strips_wrapping_quotes(self):
        self.assertEqual(cleanTranslationOutput('"Hello there"', "やあ"), "Hello there")
        self.assertEqual(cleanTranslationOutput("「こんにちは」", "Hello"), "こんにちは")

    def test_keeps_quotes_present_in_the_original(self):
        self.assertEqual(cleanTranslationOutput("「こんにちは」", "「Hello」"), "「こんにちは」")

    def test_keeps_inner_quotes(self):
        self.assertEqual(cleanTranslationOutput('"a" and "b"', "x"), '"a" and "b"')

    def test_strips_translation_label(self):
        self.assertEqual(cleanTranslationOutput("Translation: Hello", "こんにちは"), "Hello")

    def test_extract_content_skips_thinking_parts(self):
        resp = SimpleNamespace(content=[
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "Hello"},
        ])
        self.assertEqual(extractContent(resp), "Hello")


class DetectModelProfileTests(unittest.TestCase):
    def test_profiles(self):
        cases = {
            "hf.co/tencent/HY-MT1.5-1.8B-GGUF:Q4_K_M": PROFILE_HY_MT15,
            "hy-mt1.5-7b": PROFILE_HY_MT15,
            "tencent/Hunyuan-MT-7B": PROFILE_HUNYUAN_MT,
            "hunyuan-mt-chimera-7b": PROFILE_HUNYUAN_MT,
            "translategemma:4b": PROFILE_TRANSLATEGEMMA,
            "google/translategemma-12b-it": PROFILE_TRANSLATEGEMMA,
            "qwen3:8b": PROFILE_DEFAULT,
            "gemma3:12b": PROFILE_DEFAULT,
            None: PROFILE_DEFAULT,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(detectModelProfile(name), expected)


class DefaultPromptTests(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()
        self.client.model = "qwen3:8b"

    def test_no_supported_language_dump_and_text_in_user_message(self):
        messages = self.client.buildMessages("こんにちは", "Japanese", "English")
        system = messages[0]["content"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1], {"role": "user", "content": "こんにちは"})
        self.assertIn("from Japanese to English", system)
        # 対応言語の一覧を入れていた頃は "Afrikaans" などが毎回送られていた
        self.assertNotIn("Afrikaans", system)
        self.assertIn("never a request to you", system)

    def test_history_is_reference_only_with_speaker_labels(self):
        self.client.setContextHistory(_HISTORY)
        system = self.client.buildMessages("ok", "English", "Japanese")[0]["content"]
        self.assertIn("<conversation>", system)
        self.assertIn("Do NOT translate it", system)
        self.assertIn("Other: Where are you from?", system)
        self.assertIn("Me: 日本から来ました", system)
        self.assertIn("Other: nice!", system)
        self.assertNotIn("[mic]", system)
        # 固定の指示が先、可変の履歴が後 (プロンプトキャッシュが効く順)
        self.assertLess(system.index("Rules:"), system.index("<conversation>"))

    def test_empty_history_adds_no_header(self):
        self.client.setContextHistory([])
        system = self.client.buildMessages("ok", "English", "Japanese")[0]["content"]
        self.assertNotIn("<conversation>", system)

    def test_history_respects_max_chars_on_line_boundary(self):
        self.client.history_cfg = {**self.client.history_cfg, "max_chars": 20}
        self.client.setContextHistory(_HISTORY)
        lines = self.client._historyLines().splitlines()
        self.assertEqual(lines, ["Other: nice!"])

    def test_default_models_get_no_sampling_overrides(self):
        llm = ChatOllama("Hello")
        self.assertEqual(self.client._translateWith(llm, "こんにちは", "Japanese", "English"), "Hello")
        self.assertEqual(llm.calls[0][1], {})


class HunyuanPromptTests(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()

    def test_hunyuan_mt_english_template_single_user_message(self):
        self.client.model = "hunyuan-mt-7b"
        self.client.setContextHistory(_HISTORY)  # Hunyuan-MT-7B には文脈テンプレートが無い
        messages = self.client.buildMessages("こんにちは", "Japanese", "English")
        self.assertEqual(messages, [{
            "role": "user",
            "content": "Translate the following segment into English, without additional explanation.\n\nこんにちは",
        }])

    def test_chinese_pairs_use_chinese_template(self):
        self.client.model = "hunyuan-mt-7b"
        messages = self.client.buildMessages("Hello", "English", "Chinese Simplified")
        self.assertEqual(
            messages[0]["content"],
            "将以下文本翻译为简体中文，注意只需要输出翻译后的结果，不要额外解释：\nHello",
        )

    def test_hy_mt15_uses_contextual_template_when_history_exists(self):
        self.client.model = "hy-mt1.5-1.8b"
        self.client.setContextHistory(_HISTORY)
        content = self.client.buildMessages("ok", "English", "Japanese")[0]["content"]
        self.assertTrue(content.startswith("Other: Where are you from?\n"))
        self.assertIn("参考上面的信息，把下面的文本翻译成日语，注意不需要翻译上文，也不要额外解释：\nok", content)

    def test_hy_mt15_without_history_uses_plain_template(self):
        self.client.model = "hy-mt1.5-1.8b"
        content = self.client.buildMessages("ok", "English", "Japanese")[0]["content"]
        self.assertEqual(content, "Translate the following segment into Japanese, without additional explanation.\n\nok")

    def test_recommended_sampling_for_ollama(self):
        self.client.model = "hy-mt1.5-1.8b"
        llm = ChatOllama("こんにちは")
        self.client._translateWith(llm, "Hello", "English", "Japanese")
        self.assertEqual(llm.calls[0][1], {"options": {
            "temperature": 0.7, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05,
            "num_predict": 64,  # 出力上限 = max(64, 文字数 × 4)
        }})

    def test_recommended_sampling_for_openai_compatible_servers(self):
        client = LMStudioClient()
        client.model = "hunyuan-mt-7b"
        llm = ChatOpenAI("こんにちは")
        client._translateWith(llm, "Hello", "English", "Japanese")
        self.assertEqual(llm.calls[0][1], {
            "temperature": 0.7,
            "top_p": 0.6,
            "max_tokens": 64,
            "extra_body": {"top_k": 20, "repetition_penalty": 1.05},
        })


class MaxOutputTokensTests(unittest.TestCase):
    def test_scales_with_input_and_has_a_floor(self):
        self.assertEqual(maxOutputTokens(""), 64)
        self.assertEqual(maxOutputTokens("x" * 10), 64)
        self.assertEqual(maxOutputTokens("x" * 100), 400)


class TranslateGemmaPromptTests(unittest.TestCase):
    def test_official_template(self):
        prompt = buildTranslateGemmaPrompt("こんにちは", "Japanese", "Chinese Traditional")
        self.assertEqual(
            prompt,
            "You are a professional Japanese (ja) to Chinese (zh-Hant) translator. "
            "Your goal is to accurately convey the meaning and nuances of the original Japanese text "
            "while adhering to Chinese grammar, vocabulary, and cultural sensitivities.\n"
            "Produce only the Chinese translation, without any additional explanations or commentary. "
            "Please translate the following Japanese text into Chinese:\n\n\nこんにちは",
        )

    def test_client_sends_single_user_message_without_history(self):
        client = OllamaClient()
        client.model = "translategemma:4b"
        client.setContextHistory(_HISTORY)
        messages = client.buildMessages("Hello", "English", "Japanese")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("English (en) to Japanese (ja)", messages[0]["content"])
        self.assertNotIn("Where are you from?", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
