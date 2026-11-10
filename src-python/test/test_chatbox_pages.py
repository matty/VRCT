"""Pages for chatbox messages that exceed VRChat's 144-character limit."""

import unittest

from models.osc.chatbox_pages import CHATBOX_MAX_CHARS, buildStreamPages, chatboxLength


def _render(translation, message):
    """The default VRCT format: message, then the translations, one per line."""
    translation_part = "\n".join(translation)
    if translation and message:
        return message + "\n" + translation_part
    return translation_part if translation else message


def _pages(message, translation, render=_render, **kwargs):
    """Every page of a message, laid out the way the player walks through it."""
    texts = [message] + list(translation)
    offsets = [0] * len(texts)
    completed = 0
    shown = []
    for _ in range(100):
        pages = buildStreamPages(texts, offsets, completed, render, **kwargs)
        if not pages:
            return shown
        page, used = pages[0]
        shown.append(page)
        offsets = [offset + count for offset, count in zip(offsets, used)]
        completed += 1
    raise AssertionError("pages never ran out")


def _strip_marker(page):
    return page[: page.rindex(" (")]


def _line(pages, index):
    lines = [_strip_marker(page).split("\n") for page in pages]
    return "".join("".join(parts[index] for parts in lines if index < len(parts)).split())


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


class ChatboxLengthTests(unittest.TestCase):
    def test_counts_utf16_code_units(self):
        self.assertEqual(chatboxLength("abc"), 3)
        self.assertEqual(chatboxLength("こんにちは"), 5)
        # Outside the BMP: one code point, two UTF-16 code units.
        self.assertEqual(chatboxLength("😀"), 2)


class SinglePageTests(unittest.TestCase):
    def test_message_that_fits_is_returned_exactly_as_rendered(self):
        self.assertEqual(_pages("hello", ["hola"]), ["hello\nhola"])

    def test_message_exactly_at_the_limit_is_not_split(self):
        message = "a" * CHATBOX_MAX_CHARS
        self.assertEqual(_pages(message, []), [message])

    def test_empty_translation_list(self):
        self.assertEqual(_pages("hello", []), ["hello"])

    def test_translation_only(self):
        self.assertEqual(_pages("", ["hola"]), ["hola"])

    def test_nothing_to_show(self):
        self.assertEqual(buildStreamPages(["", ""], [0, 0], 0, _render), [])
        self.assertEqual(buildStreamPages(["abc"], [3], 1, _render), [])


class LongMessageTests(unittest.TestCase):
    def test_every_page_fits_the_limit(self):
        pages = _pages(MESSAGE, [TRANSLATION])
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)

    def test_pages_are_numbered(self):
        pages = _pages(MESSAGE, [TRANSLATION])
        for index, page in enumerate(pages, start=1):
            self.assertTrue(page.endswith(f" ({index}/{len(pages)})"), page)

    def test_nothing_is_lost(self):
        pages = _pages(MESSAGE, [TRANSLATION])
        self.assertEqual(_line(pages, 0), "".join(MESSAGE.split()))
        self.assertEqual(_line(pages, 1), TRANSLATION)

    def test_pages_end_on_the_same_sentence_in_original_and_translation(self):
        pages = _pages(MESSAGE, [TRANSLATION])
        original, translation = _strip_marker(pages[0]).split("\n")
        self.assertTrue(original.startswith("Hello, my name is Matty"))
        self.assertTrue(original.endswith("."))
        self.assertTrue(translation.endswith("。"))
        self.assertEqual(original.count(". ") + 1, translation.count("。"))

    def test_translation_is_no_longer_cut_off(self):
        # What VRChat does to the unsplit message: keeps the first 144
        # characters, which drops the end of the translation.
        truncated = _render([TRANSLATION], MESSAGE)[:CHATBOX_MAX_CHARS]
        self.assertNotIn("ほぼ毎晩ここに来ます。", truncated)
        self.assertIn("ほぼ毎晩ここに来ます。", "".join(_pages(MESSAGE, [TRANSLATION])))

    def test_several_target_languages(self):
        translations = [
            TRANSLATION,
            "안녕하세요, 제 이름은 매티이고 영국 출신입니다. VRChat을 시작한 지 약 3년이 되었습니다. "
            "이 세계는 정말 아름답네요. 거의 매일 밤 여기에 옵니다.",
            "你好，我叫马蒂，来自英国。我玩VRChat大约三年了。这个世界真的很美，不是吗？我几乎每晚都来这里。",
        ]
        pages = _pages(MESSAGE, translations)
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)
        for index, text in enumerate([MESSAGE] + translations):
            self.assertEqual(_line(pages, index), "".join(text.split()))

    def test_text_without_sentence_ends_is_cut_at_spaces(self):
        message = " ".join(["word"] * 80)
        pages = _pages(message, [])
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)
            self.assertTrue(set(_strip_marker(page).split(" ")) <= {"word"}, page)
        self.assertEqual(" ".join(_strip_marker(page) for page in pages), message)

    def test_text_without_spaces_is_cut_anywhere(self):
        message = "あ" * 400
        pages = _pages(message, [])
        self.assertEqual("".join(_strip_marker(page) for page in pages), message)
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)

    def test_short_original_with_long_translation(self):
        translation = " ".join(["translated"] * 40)
        pages = _pages("はい", [translation])
        self.assertGreater(len(pages), 1)
        self.assertTrue(pages[0].startswith("はい\n"))
        for page in pages[1:]:
            self.assertNotIn("はい", page)
        self.assertEqual(" ".join(_strip_marker(page).split("\n")[-1] for page in pages), translation)

    def test_respects_the_users_format(self):
        def render(translation, message):
            return f"[{message}] => <{' | '.join(translation)}>"

        pages = _pages(MESSAGE, [TRANSLATION], render)
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertTrue(page.startswith("["), page)
            self.assertIn("] => <", page)
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)

    def test_astral_characters_count_as_two(self):
        message = "😀" * 100  # 200 UTF-16 code units, 100 code points
        pages = _pages(message, [])
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)
        self.assertEqual("".join(_strip_marker(page) for page in pages), message)

    def test_format_with_long_fixed_text_still_gets_through(self):
        def render(translation, message):
            return "#" * 100 + message + "".join(translation)

        pages = _pages("hello there, this is a long message " * 3, ["translated text " * 5], render)
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)

    def test_decimals_are_not_treated_as_sentence_ends(self):
        message = "The price went up by 3.5 percent. " * 6
        for page in _pages(message.strip(), []):
            self.assertNotRegex(_strip_marker(page), r"3\.$")


class GrowingMessageTests(unittest.TestCase):
    """Pages are filled front to back, so growth never moves a full page."""

    def test_full_pages_do_not_change_when_the_text_grows(self):
        sentences = MESSAGE.split(". ")
        translated = TRANSLATION.split("。")[:-1]
        first_page = None
        for upto in range(3, len(sentences) + 1):
            message = ". ".join(sentences[:upto])
            translation = "。".join(translated[:upto]) + "。"
            pages = buildStreamPages([message, translation], [0, 0], 0, _render)
            if len(pages) < 2:
                continue
            body = _strip_marker(pages[0][0])
            if first_page is None:
                first_page = body
            self.assertEqual(body, first_page)
        self.assertIsNotNone(first_page)

    def test_consumed_counts_cover_the_whole_text(self):
        texts = [MESSAGE, TRANSLATION]
        pages = buildStreamPages(texts, [0, 0], 0, _render)
        self.assertEqual([sum(used[i] for _, used in pages) for i in range(2)], [len(MESSAGE), len(TRANSLATION)])

    def test_resuming_from_offsets_gives_the_following_pages(self):
        texts = [MESSAGE, TRANSLATION]
        pages = buildStreamPages(texts, [0, 0], 0, _render)
        rest = buildStreamPages(texts, pages[0][1], 1, _render)
        self.assertEqual([page for page, _ in rest], [page for page, _ in pages[1:]])

    def test_lookahead_is_limited(self):
        pages = buildStreamPages(["word " * 2000], [0], 0, _render, max_pages=5)
        self.assertEqual(len(pages), 5)


if __name__ == "__main__":
    unittest.main()
