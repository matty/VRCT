"""Chat laughter ("www", "草", "ㅋㅋㅋ", "lol") for the local MT models.

NLLB/M2M100 invent sentences for laughter ("それはやばいwww" came out as
profanity), so it is taken out before translating and written back in the
target language's own convention.
"""

import unittest

from models.translation.translation_laughter import joinLaughter, laughterFor, splitLaughter
from models.translation.translation_translator import Translator

NLLB = "nllb-200-distilled-600M-ct2-int8"
L = (True, "")


def _text(text):
    return (False, text)


class SplitLaughterTests(unittest.TestCase):
    def _assertSplit(self, language, cases):
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(splitLaughter(message, language), expected)

    def test_japanese(self):
        self._assertSplit("Japanese", {
            "www": [L],
            "ｗｗｗ": [L],
            "ww": [L],
            "草": [L],
            "草草草": [L],
            "大草原": [L],
            "草生える": [L],
            "ワロタ": [L],
            "ははは": [L],
            "草www": [L],
            "www！！": [L],
            "それなwww": [_text("それな"), L],
            "まじかw": [_text("まじか"), L],
            "それは草": [_text("それは"), L],
            "それは草ｗ": [_text("それは"), L],
            "まじ草": [_text("まじ"), L],
            "今日めっちゃ楽しかった笑": [_text("今日めっちゃ楽しかった"), L],
            "なにそれ（笑）": [_text("なにそれ"), L],
            "なにそれ(笑)": [_text("なにそれ"), L],
            "まじかwww それはやばいwww": [_text("まじか"), L, _text("それはやばい"), L],
            "いやwwwそれはないwww": [_text("いや"), L, _text("それはない"), L],
            "あはは、そうだね": [L, _text("そうだね")],
        })

    def test_japanese_text_that_only_looks_like_laughter(self):
        for message in (
            "庭の草を刈った",          # grass
            "草を食べる",
            "wwwにアクセスする",       # the web
            "www.example.com を見て",
            "爆笑した",                # a word that contains 笑
            "笑うしかない",
            "彼は笑った",
            "今日は5wの電球を買った",   # watts
            "wifiが遅い",
            "wow すごい",
            "かわいい！",
        ):
            with self.subTest(message=message):
                self.assertEqual(splitLaughter(message, "Japanese"), [_text(message)])

    def test_korean(self):
        self._assertSplit("Korean", {
            "ㅋㅋㅋ": [L],
            "ㅎㅎ": [L],
            "진짜 웃기다 ㅋㅋㅋㅋ": [_text("진짜 웃기다"), L],
            "웃겨ㅋ": [_text("웃겨"), L],
            "안녕하세요": [_text("안녕하세요")],
        })

    def test_chinese(self):
        for language in ("Chinese Simplified", "Chinese Traditional"):
            self._assertSplit(language, {
                "哈哈哈": [L],
                "呵呵": [L],
                "笑死我了": [L],
                "2333": [L],
                "太好笑了哈哈哈哈": [_text("太好笑了"), L],
                "我有2333个": [_text("我有2333个")],
                "他笑死了很多人": [_text("他笑死了很多人")],
                "他哈哈大笑": [_text("他哈哈大笑")],   # an idiom, not chat laughter
                "她笑嘻嘻的": [_text("她笑嘻嘻的")],
            })

    def test_english(self):
        self._assertSplit("English", {
            "lol": [L],
            "LOL": [L],
            "lmao": [L],
            "hahaha": [L],
            "that's so funny lol": [_text("that's so funny"), L],
            "no way lmao": [_text("no way"), L],
            "lol what": [L, _text("what")],
            "haha nice": [L, _text("nice")],
            "that was great, lol. see you": [_text("that was great,"), L, _text("see you")],
        })

    def test_english_words_that_only_look_like_laughter(self):
        for message in (
            "I play lol every day",  # the game: not at the edge of a clause
            "hello",
            "aha I see",
            "he he said",
            "shh",
        ):
            with self.subTest(message=message):
                self.assertEqual(splitLaughter(message, "English"), [_text(message)])

    def test_other_languages(self):
        self.assertEqual(splitLaughter("qué gracioso jajaja", "Spanish"), [_text("qué gracioso"), L])
        self.assertEqual(splitLaughter("muito engraçado kkkk", "Portuguese"), [_text("muito engraçado"), L])
        self.assertEqual(splitLaughter("ตลกมาก 5555", "Thai"), [_text("ตลกมาก"), L])
        self.assertEqual(splitLaughter("ราคา 555 บาท", "Thai"), [_text("ราคา 555 บาท")])
        self.assertEqual(splitLaughter("очень смешно ахаха", "Russian"), [_text("очень смешно"), L])
        self.assertEqual(splitLaughter("lucu banget wkwkwk", "Indonesian"), [_text("lucu banget"), L])
        self.assertEqual(splitLaughter("trop drôle mdr", "French"), [_text("trop drôle"), L])

    def test_haha_is_recognised_in_every_language(self):
        self.assertEqual(splitLaughter("das ist lustig haha", "German"), [_text("das ist lustig"), L])
        self.assertEqual(splitLaughter("hahaha", "Swahili"), [L])

    def test_message_without_laughter_is_returned_unchanged(self):
        message = "  Hello there.  How are you?  "
        self.assertEqual(splitLaughter(message, "English"), [_text(message)])

    def test_laughter_of_another_language_is_left_to_the_model(self):
        # ㅋㅋㅋ typed while the source language is English is not recognised.
        self.assertEqual(splitLaughter("ㅋㅋㅋ", "English"), [_text("ㅋㅋㅋ")])


class JoinLaughterTests(unittest.TestCase):
    def test_each_language_gets_its_own_laughter(self):
        self.assertEqual(laughterFor("English"), "lol")
        self.assertEqual(laughterFor("Japanese"), "笑")
        self.assertEqual(laughterFor("Korean"), "ㅋㅋㅋ")
        self.assertEqual(laughterFor("Chinese Simplified"), "哈哈哈")
        self.assertEqual(laughterFor("Spanish"), "jajaja")
        self.assertEqual(laughterFor("German"), "haha")

    def test_laughter_only(self):
        self.assertEqual(joinLaughter([L], [], "English"), "lol")
        self.assertEqual(joinLaughter([L], [], "Japanese"), "笑")

    def test_laughter_follows_the_sentence(self):
        pieces = [_text("それはやばい"), L]
        self.assertEqual(joinLaughter(pieces, ["That's crazy"], "English"), "That's crazy lol")
        self.assertEqual(joinLaughter(pieces, ["그건 대박이야"], "Korean"), "그건 대박이야 ㅋㅋㅋ")

    def test_no_space_before_laughter_in_japanese_and_chinese(self):
        pieces = [_text("that's so funny"), L]
        self.assertEqual(joinLaughter(pieces, ["面白かった"], "Japanese"), "面白かった笑")
        self.assertEqual(joinLaughter(pieces, ["太好笑了"], "Chinese Simplified"), "太好笑了哈哈哈")

    def test_full_stop_before_laughter_is_dropped(self):
        pieces = [_text("x"), L]
        self.assertEqual(joinLaughter(pieces, ["That's crazy."], "English"), "That's crazy lol")
        self.assertEqual(joinLaughter(pieces, ["面白かった。"], "Japanese"), "面白かった笑")

    def test_other_sentence_endings_are_kept(self):
        pieces = [_text("x"), L]
        self.assertEqual(joinLaughter(pieces, ["What's that?"], "English"), "What's that? lol")
        self.assertEqual(joinLaughter(pieces, ["Well..."], "English"), "Well... lol")

    def test_keeps_the_order_of_the_message(self):
        pieces = [_text("a"), L, _text("b"), L]
        self.assertEqual(joinLaughter(pieces, ["No way", "That's crazy"], "English"), "No way lol That's crazy lol")
        self.assertEqual(joinLaughter([L, _text("a")], ["what"], "English"), "lol what")


class _Result:
    def __init__(self, tokens):
        self.hypotheses = [tokens]


class _Tokenizer:
    """One character = one token."""

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
    """Returns "[<target>]<source>" as the translation."""

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


class CTranslate2LaughterTests(unittest.TestCase):
    def test_laughter_is_not_sent_to_the_model(self):
        translator = _loadedTranslator()
        result = translator.translate("CTranslate2", NLLB, "Japanese", "English", "United States", "それはやばいwww")
        sources, prefixes, _ = translator.ctranslate2_translator.calls[0]
        self.assertEqual(sources, [list("それはやばい")])
        self.assertEqual(prefixes, [["eng_Latn"]])
        self.assertEqual(result, "[eng_Latn]それはやばい lol")

    def test_laughter_only_message_does_not_call_the_model(self):
        translator = _loadedTranslator()
        result = translator.translate("CTranslate2", NLLB, "Japanese", "English", "United States", "草")
        self.assertEqual(result, "lol")
        self.assertEqual(translator.ctranslate2_translator.calls, [])

    def test_laughter_only_message_works_without_a_loaded_model(self):
        translator = Translator()
        self.assertFalse(translator.isLoadedCTranslate2Model())
        result = translator.translate("CTranslate2", NLLB, "Korean", "Japanese", "Japan", "ㅋㅋㅋㅋ")
        self.assertEqual(result, "笑")

    def test_text_between_laughter_is_translated_in_one_batch(self):
        translator = _loadedTranslator()
        result = translator.translate(
            "CTranslate2", NLLB, "Japanese", "English", "United States", "まじかwww それはやばいwww"
        )
        self.assertEqual(len(translator.ctranslate2_translator.calls), 1)
        sources, _, _ = translator.ctranslate2_translator.calls[0]
        self.assertEqual(sources, [list("まじか"), list("それはやばい")])
        self.assertEqual(result, "[eng_Latn]まじか lol [eng_Latn]それはやばい lol")

    def test_each_target_language_gets_its_own_laughter(self):
        translator = _loadedTranslator()
        results = translator.translateCTranslate2Targets(
            NLLB,
            "Japanese",
            [("English", "United States"), ("Korean", "South Korea"), ("Chinese Simplified", "China")],
            "楽しかった笑",
        )
        self.assertEqual(len(translator.ctranslate2_translator.calls), 1)
        self.assertEqual(
            results,
            ["[eng_Latn]楽しかった lol", "[kor_Hang]楽しかった ㅋㅋㅋ", "[zho_Hans]楽しかった哈哈哈"],
        )

    def test_multi_target_laughter_only(self):
        translator = _loadedTranslator()
        results = translator.translateCTranslate2Targets(
            NLLB, "Japanese", [("English", "United States"), ("Korean", "South Korea")], "www",
        )
        self.assertEqual(results, ["lol", "ㅋㅋㅋ"])
        self.assertEqual(translator.ctranslate2_translator.calls, [])

    def test_same_language_target_is_returned_untouched(self):
        translator = _loadedTranslator()
        results = translator.translateCTranslate2Targets(
            NLLB, "Japanese", [("Japanese", "Japan"), ("English", "United States")], "草",
        )
        self.assertEqual(results, ["草", "lol"])

    def test_message_without_laughter_is_translated_as_before(self):
        translator = _loadedTranslator()
        result = translator.translate(
            "CTranslate2", NLLB, "English", "Japanese", "Japan", "Hello there friend. See you again tomorrow."
        )
        sources, _, _ = translator.ctranslate2_translator.calls[0]
        self.assertEqual(sources, [list("Hello there friend."), list("See you again tomorrow.")])
        self.assertEqual(result, "[jpn_Jpan]Hello there friend. [jpn_Jpan]See you again tomorrow.")

    def test_sentences_are_still_split_inside_a_text_piece(self):
        translator = _loadedTranslator()
        result = translator.translate(
            "CTranslate2", NLLB, "English", "Japanese", "Japan", "Hello there friend. See you again tomorrow lol"
        )
        sources, _, _ = translator.ctranslate2_translator.calls[0]
        self.assertEqual(sources, [list("Hello there friend."), list("See you again tomorrow")])
        self.assertEqual(result, "[jpn_Jpan]Hello there friend. [jpn_Jpan]See you again tomorrow笑")

    def test_result_with_laughter_is_cached(self):
        translator = _loadedTranslator()
        first = translator.translate("CTranslate2", NLLB, "Japanese", "English", "United States", "それなwww")
        second = translator.translate("CTranslate2", NLLB, "Japanese", "English", "United States", "それなwww")
        self.assertEqual(first, second)
        self.assertEqual(len(translator.ctranslate2_translator.calls), 1)

    def test_direct_batch_call_without_language_names_does_not_touch_laughter(self):
        # Callers that only have language codes keep the previous behaviour.
        translator = _loadedTranslator()
        result = translator.translateCTranslate2("それなwww", "jpn_Jpan", "eng_Latn", NLLB)
        sources, _, _ = translator.ctranslate2_translator.calls[0]
        self.assertEqual(sources, [list("それなwww")])
        self.assertEqual(result, "[eng_Latn]それなwww")

    def test_other_engines_are_not_affected(self):
        import models.translation.translation_translator as tt
        from unittest.mock import MagicMock, patch

        translator = Translator()
        web = MagicMock(return_value="translated")
        with patch.object(tt, "other_web_Translator", web), patch.object(tt, "ENABLE_TRANSLATORS", True):
            translator.translate("Google", NLLB, "Japanese", "English", "United States", "それなwww")
        self.assertEqual(web.call_args.kwargs["query_text"], "それなwww")


if __name__ == "__main__":
    unittest.main()
