"""CTranslate2 の文分割・一括翻訳と、翻訳キャッシュのテスト。

- NLLB/M2M100 は文単位のモデルなので、複数文は文ごとに訳してつなぐ
- 複数ターゲット言語は translate_batch 1 回にまとめる (ロックで直列化される
  ため、スレッドで並列に投げても速くならなかった)
- 幻覚による繰り返しで延々とデコードしないよう出力長に上限を付ける
- 決定的なエンジン (Google/Bing/Papago/CTranslate2) の結果はキャッシュする
"""

import unittest
from unittest.mock import MagicMock, patch

import models.translation.translation_translator as tt
from models.translation.translation_translator import Translator, joinSentences, mergeShortSentences, splitSentences

NLLB = "nllb-200-distilled-600M-ct2-int8"


class _Result:
    def __init__(self, tokens):
        self.hypotheses = [tokens]


class _Tokenizer:
    """1 文字 = 1 トークン。decode は "<lang>:<text>" を返す。"""

    lang_code_to_token = {"en": "__en__", "ja": "__ja__"}

    def __init__(self):
        self.src_lang = None

    def encode(self, message):
        return list(message)

    def convert_ids_to_tokens(self, ids):
        return ids

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def decode(self, ids):
        return "".join(ids)


class _CT2:
    """入力をそのまま「訳文」として返す。prefix を先頭に付けて返すのは本物と同じ。"""

    def __init__(self):
        self.calls = []

    def translate_batch(self, sources, target_prefix, **kwargs):
        self.calls.append((sources, target_prefix, kwargs))
        return [_Result(prefix + [f"[{prefix[0]}]"] + src) for src, prefix in zip(sources, target_prefix)]


def _loadedTranslator():
    translator = Translator()
    translator.is_loaded_ctranslate2_model = True
    translator.ctranslate2_tokenizer = _Tokenizer()
    translator.ctranslate2_translator = _CT2()
    return translator


class SentenceSplitTests(unittest.TestCase):
    def test_splits(self):
        cases = {
            "Hello there. How are you? I am fine!": ["Hello there.", "How are you?", "I am fine!"],
            "Mr. Smith is here. It costs 3.5 dollars.": ["Mr. Smith is here.", "It costs 3.5 dollars."],
            "U.S. troops left. Then what": ["U.S. troops left.", "Then what"],
            '"Really?" she said. Yes.': ['"Really?" she said.', "Yes."],
            "こんにちは。元気ですか？はい！": ["こんにちは。", "元気ですか？", "はい！"],
            "Hi! 😊": ["Hi! 😊"],
            "No. I don't think so.": ["No.", "I don't think so."],
            "first line\nsecond line": ["first line", "second line"],
            "ok": ["ok"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(splitSentences(text), expected)

    def test_join(self):
        self.assertEqual(joinSentences(["Hello.", "How are you?"]), "Hello. How are you?")
        self.assertEqual(joinSentences(["こんにちは。", "元気ですか？"]), "こんにちは。元気ですか？")
        # NLLB は日本語の訳に「。」を付けないことが多い。そのときは空白で区切る
        self.assertEqual(joinSentences(["すみません", "行かないと"]), "すみません 行かないと")
        self.assertEqual(joinSentences(["Wait…", "What?"]), "Wait… What?")


class MergeShortSentencesTests(unittest.TestCase):
    def test_merge(self):
        self.assertEqual(mergeShortSentences(["a", "b", "c"], [12, 12, 12]), ["a", "b", "c"])
        self.assertEqual(mergeShortSentences(["Hi!", "long one"], [3, 12]), ["Hi! long one"])
        self.assertEqual(mergeShortSentences(["long one", "Bye!"], [12, 3]), ["long one Bye!"])
        self.assertEqual(mergeShortSentences(["はじめまして！", "日本から来ました。"], [5, 8]), ["はじめまして！日本から来ました。"])


class CTranslate2BatchTests(unittest.TestCase):
    def test_multi_sentence_translates_each_sentence_and_joins(self):
        translator = _loadedTranslator()
        result = translator.translateCTranslate2("Hello there friend. See you again tomorrow.", "eng_Latn", "jpn_Jpan", NLLB)
        sources, prefixes, _ = translator.ctranslate2_translator.calls[0]
        self.assertEqual(sources, [list("Hello there friend."), list("See you again tomorrow.")])
        self.assertEqual(prefixes, [["jpn_Jpan"], ["jpn_Jpan"]])
        self.assertEqual(result, "[jpn_Jpan]Hello there friend. [jpn_Jpan]See you again tomorrow.")

    def test_short_sentence_is_translated_together_with_its_neighbour(self):
        translator = _loadedTranslator()
        translator.translateCTranslate2("Hi. See you again tomorrow.", "eng_Latn", "jpn_Jpan", NLLB)
        sources, _, _ = translator.ctranslate2_translator.calls[0]
        self.assertEqual(sources, [list("Hi. See you again tomorrow.")])

    def test_several_targets_go_into_one_batch_in_slot_order(self):
        translator = _loadedTranslator()
        results = translator.translateCTranslate2Batch("Hello there friend. See you again tomorrow.", "eng_Latn", ["jpn_Jpan", "kor_Hang"], NLLB)
        self.assertEqual(len(translator.ctranslate2_translator.calls), 1)
        _, prefixes, _ = translator.ctranslate2_translator.calls[0]
        self.assertEqual(prefixes, [["jpn_Jpan"], ["jpn_Jpan"], ["kor_Hang"], ["kor_Hang"]])
        self.assertEqual(results, ["[jpn_Jpan]Hello there friend. [jpn_Jpan]See you again tomorrow.", "[kor_Hang]Hello there friend. [kor_Hang]See you again tomorrow."])

    def test_decoding_length_is_bounded_by_input_length(self):
        translator = _loadedTranslator()
        translator.translateCTranslate2("hello", "eng_Latn", "jpn_Jpan", NLLB)
        _, _, kwargs = translator.ctranslate2_translator.calls[0]
        self.assertEqual(kwargs["max_decoding_length"], 5 * 3 + 16)

        translator.translateCTranslate2("x" * 200, "eng_Latn", "jpn_Jpan", NLLB)
        _, _, kwargs = translator.ctranslate2_translator.calls[1]
        self.assertEqual(kwargs["max_decoding_length"], 256)

    def test_failure_returns_false_for_every_target(self):
        translator = _loadedTranslator()
        translator.ctranslate2_translator.translate_batch = MagicMock(side_effect=RuntimeError("boom"))
        with patch.object(tt, "errorLogging"):
            results = translator.translateCTranslate2Batch("hi", "eng_Latn", ["jpn_Jpan", "kor_Hang"], NLLB)
        self.assertEqual(results, [False, False])

    def test_unloaded_model_returns_false(self):
        translator = Translator()
        self.assertEqual(translator.translateCTranslate2Batch("hi", "eng_Latn", ["jpn_Jpan"], NLLB), [False])


class CTranslate2TargetsTests(unittest.TestCase):
    def test_maps_languages_skips_same_language_and_unsupported(self):
        translator = _loadedTranslator()
        results = translator.translateCTranslate2Targets(
            NLLB, "English",
            [("Japanese", "Japan"), ("English", "United States"), ("Klingon", "Qo'noS")],
            "Hi",
        )
        self.assertEqual(results, ["[jpn_Jpan]Hi", "Hi", None])

    def test_second_call_is_served_from_cache(self):
        translator = _loadedTranslator()
        targets = [("Japanese", "Japan"), ("Korean", "South Korea")]
        first = translator.translateCTranslate2Targets(NLLB, "English", targets, "Hi")
        second = translator.translateCTranslate2Targets(NLLB, "English", targets, "Hi")
        self.assertEqual(first, second)
        self.assertEqual(len(translator.ctranslate2_translator.calls), 1)

    def test_single_translate_shares_the_cache(self):
        translator = _loadedTranslator()
        translator.translateCTranslate2Targets(NLLB, "English", [("Japanese", "Japan")], "Hi")
        result = translator.translate("CTranslate2", NLLB, "English", "Japanese", "Japan", "Hi")
        self.assertEqual(result, "[jpn_Jpan]Hi")
        self.assertEqual(len(translator.ctranslate2_translator.calls), 1)


class TranslateCacheAndContextTests(unittest.TestCase):
    def test_web_engine_result_is_cached(self):
        translator = Translator()
        web = MagicMock(return_value="Hello")
        with patch.object(tt, "other_web_Translator", web), patch.object(tt, "ENABLE_TRANSLATORS", True):
            self.assertEqual(translator.translate("Google", NLLB, "Japanese", "English", "Japan", "こんにちは"), "Hello")
            self.assertEqual(translator.translate("Google", NLLB, "Japanese", "English", "United States", "こんにちは"), "Hello")
        self.assertEqual(web.call_count, 1)

    def test_empty_result_is_a_failure_and_not_cached(self):
        translator = Translator()
        web = MagicMock(return_value="  ")
        with patch.object(tt, "other_web_Translator", web), patch.object(tt, "ENABLE_TRANSLATORS", True):
            self.assertIs(translator.translate("Google", NLLB, "Japanese", "English", "Japan", "こんにちは"), False)
            self.assertIs(translator.translate("Google", NLLB, "Japanese", "English", "Japan", "こんにちは"), False)
        self.assertEqual(web.call_count, 2)

    def test_llm_engines_are_not_cached(self):
        translator = Translator()
        client = MagicMock()
        client.translate.return_value = "Hello"
        translator._provider_clients["Gemini_API"] = client
        for _ in range(2):
            translator.translate("Gemini_API", NLLB, "Japanese", "English", "Japan", "こんにちは")
        self.assertEqual(client.translate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
