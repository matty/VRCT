"""LLM 翻訳のプロンプト組み立てと出力の後処理 (translation_llm_base) のテスト。

- 既定プロンプトに対応言語の一覧 (数百トークン) を入れない
- 会話履歴は「参照専用」と明示し、発話者を Me/Other で渡す
- 推論モデルの <think> や全体を囲む引用符を出力から取り除く
"""

import unittest
from types import SimpleNamespace

from models.translation.translation_llm_base import (
    cleanTranslationOutput,
    extractContent,
)
from models.translation.translation_ollama import OllamaClient


class ChatOllama:  # 型名で呼び出し方を切り替えるので、名前だけ本物に合わせる
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content=self.reply)



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


if __name__ == "__main__":
    unittest.main()
