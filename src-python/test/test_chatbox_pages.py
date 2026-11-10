"""Splitting chatbox messages that exceed VRChat's 144-character limit."""

import unittest

from models.osc.chatbox_pages import (
    CHATBOX_MAX_CHARS,
    CHATBOX_MAX_PAGES,
    buildChatboxPages,
    chatboxLength,
    splitText,
)


def _render(translation, message, separator="\n"):
    """The default VRCT format: message, then the translations, one per line."""
    translation_part = "\n".join(translation)
    if translation and message:
        return message + separator + translation_part
    return translation_part if translation else message


def _strip_marker(page):
    return page[: page.rindex(" (")]


class ChatboxLengthTests(unittest.TestCase):
    def test_counts_utf16_code_units(self):
        self.assertEqual(chatboxLength("abc"), 3)
        self.assertEqual(chatboxLength("こんにちは"), 5)
        # Outside the BMP: one code point, two UTF-16 code units.
        self.assertEqual(chatboxLength("😀"), 2)


class SinglePageTests(unittest.TestCase):
    def test_message_that_fits_is_returned_exactly_as_rendered(self):
        pages = buildChatboxPages("hello", ["hola"], _render)
        self.assertEqual(pages, ["hello\nhola"])

    def test_message_exactly_at_the_limit_is_not_split(self):
        message = "a" * CHATBOX_MAX_CHARS
        self.assertEqual(buildChatboxPages(message, [], _render), [message])

    def test_empty_translation_list(self):
        self.assertEqual(buildChatboxPages("hello", [], _render), ["hello"])

    def test_translation_only(self):
        self.assertEqual(buildChatboxPages("", ["hola"], _render), ["hola"])


class SplitTests(unittest.TestCase):
    MESSAGE = (
        "Hello, my name is Matty and I'm from England. "
        "I have been playing VRChat for about three years now. "
        "This world is really beautiful, isn't it? "
        "I come here almost every night."
    )
    TRANSLATION = (
        "こんにちは、私の名前はマティで、イギリス出身です。"
        "VRChatを始めて約3年になります。"
        "この世界は本当に美しいですね。"
        "ほぼ毎晩ここに来ます。"
    )

    def test_every_page_fits_the_limit(self):
        pages = buildChatboxPages(self.MESSAGE, [self.TRANSLATION], _render)
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)

    def test_pages_are_numbered(self):
        pages = buildChatboxPages(self.MESSAGE, [self.TRANSLATION], _render)
        for index, page in enumerate(pages, start=1):
            self.assertTrue(page.endswith(f" ({index}/{len(pages)})"), page)

    def test_nothing_is_lost(self):
        pages = buildChatboxPages(self.MESSAGE, [self.TRANSLATION], _render)
        originals = []
        translations = []
        for page in pages:
            original, translation = _strip_marker(page).split("\n")
            originals.append(original)
            translations.append(translation)
        self.assertEqual(" ".join(originals), self.MESSAGE)
        self.assertEqual("".join(translations), self.TRANSLATION)

    def test_each_page_keeps_the_original_with_its_translation(self):
        # Same number of sentences on both sides: pages are cut on the same
        # sentence boundaries, so the lines of a page say the same thing.
        pages = buildChatboxPages(self.MESSAGE, [self.TRANSLATION], _render)
        first_original, first_translation = _strip_marker(pages[0]).split("\n")
        self.assertTrue(first_original.startswith("Hello, my name is Matty"))
        self.assertTrue(first_original.endswith("England."))
        self.assertEqual(first_translation, "こんにちは、私の名前はマティで、イギリス出身です。")
        last_original, last_translation = _strip_marker(pages[-1]).split("\n")
        self.assertTrue(last_original.endswith("I come here almost every night."))
        self.assertTrue(last_translation.endswith("ほぼ毎晩ここに来ます。"))

    def test_translation_is_no_longer_cut_off(self):
        # What VRChat does to the unsplit message: keeps the first 144
        # characters, which drops the end of the translation.
        truncated = _render([self.TRANSLATION], self.MESSAGE)[:CHATBOX_MAX_CHARS]
        self.assertNotIn("ほぼ毎晩ここに来ます。", truncated)
        pages = buildChatboxPages(self.MESSAGE, [self.TRANSLATION], _render)
        self.assertIn("ほぼ毎晩ここに来ます。", "".join(pages))

    def test_several_target_languages(self):
        translations = [
            self.TRANSLATION,
            "안녕하세요, 제 이름은 매티이고 영국 출신입니다. VRChat을 시작한 지 약 3년이 되었습니다. "
            "이 세계는 정말 아름답네요. 거의 매일 밤 여기에 옵니다.",
            "你好，我叫马蒂，来自英国。我玩VRChat大约三年了。这个世界真的很美，不是吗？我几乎每晚都来这里。",
        ]
        pages = buildChatboxPages(self.MESSAGE, translations, _render)
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)
            # original + three translations on every page
            self.assertEqual(_strip_marker(page).count("\n"), 3)

    def test_text_without_sentence_ends_is_cut_at_spaces(self):
        message = " ".join(["word"] * 80)
        pages = buildChatboxPages(message, [], _render)
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)
            self.assertTrue(set(_strip_marker(page).split(" ")) <= {"word"}, page)
        self.assertEqual(" ".join(_strip_marker(page) for page in pages), message)

    def test_text_without_spaces_is_cut_anywhere(self):
        message = "あ" * 400
        pages = buildChatboxPages(message, [], _render)
        self.assertEqual("".join(_strip_marker(page) for page in pages), message)
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)

    def test_short_original_with_long_translation(self):
        # The original cannot be split into as many parts as the translation:
        # it goes on the first page, the rest are translation only.
        translation = " ".join(["translated"] * 40)
        pages = buildChatboxPages("はい", [translation], _render)
        self.assertGreater(len(pages), 1)
        self.assertTrue(pages[0].startswith("はい\n"))
        for page in pages[1:]:
            self.assertNotIn("\n", page)
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)

    def test_respects_the_users_format(self):
        def render(translation, message):
            return f"[{message}] => <{' | '.join(translation)}>"

        pages = buildChatboxPages(self.MESSAGE, [self.TRANSLATION], render)
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertTrue(page.startswith("["), page)
            self.assertIn("] => <", page)
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)

    def test_astral_characters_count_as_two(self):
        message = "😀" * 100  # 200 UTF-16 code units, 100 code points
        pages = buildChatboxPages(message, [], _render)
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)

    def test_format_that_can_never_fit_falls_back_to_fixed_size_pieces(self):
        def render(translation, message):
            return "#" * 200 + message

        pages = buildChatboxPages("hello", [], render)
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)

    def test_huge_message_is_capped(self):
        pages = buildChatboxPages("x" * 5000, [], _render)
        self.assertEqual(len(pages), CHATBOX_MAX_PAGES)
        self.assertIn("…", pages[-1])
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)


class SplitTextTests(unittest.TestCase):
    def test_prefers_sentence_ends(self):
        self.assertEqual(
            splitText("One two three four. Five six seven eight nine.", 2),
            ["One two three four.", "Five six seven eight nine."],
        )

    def test_does_not_split_decimals(self):
        pieces = splitText("The price went up by 3.5 percent over the last few weeks they said", 2)
        self.assertTrue(any("3.5" in piece for piece in pieces), pieces)

    def test_too_short_to_split(self):
        self.assertEqual(splitText("ok", 3), ["ok", "", ""])

    def test_single_part(self):
        self.assertEqual(splitText("  hello  ", 1), ["hello"])


if __name__ == "__main__":
    unittest.main()
