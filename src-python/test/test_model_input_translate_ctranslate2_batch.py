"""getInputTranslate が CTranslate2 の複数ターゲットを 1 回のバッチで訳すことのテスト。

バッチで訳せなかったターゲットだけ従来の getTranslate() (フォールバック/リトライ
付き) に回す。
"""

import unittest
from unittest.mock import MagicMock

from model import Model, config


class TestInputTranslateCTranslate2Batch(unittest.TestCase):
    def setUp(self) -> None:
        self.model = object.__new__(Model)
        self.model._inited = True
        self.model._init_failed = False
        self.model.translator = MagicMock()
        self.model._translation_executor = MagicMock()  # 使われないこと

        self._saved = (config.SELECTED_TARGET_LANGUAGES, config.SELECTED_TRANSLATION_ENGINES)
        tab = config.SELECTED_TAB_NO
        engines = dict(config.SELECTED_TRANSLATION_ENGINES)
        engines[tab] = "CTranslate2"
        config.SELECTED_TRANSLATION_ENGINES = engines
        targets = config.SELECTED_TARGET_LANGUAGES
        for no, (language, country) in zip(("1", "2", "3"), (
            ("English", "United States"), ("Korean", "South Korea"), ("French", "France"),
        )):
            targets[tab][no].update({"language": language, "country": country, "enable": True})
        config.SELECTED_TARGET_LANGUAGES = targets

    def tearDown(self) -> None:
        config.SELECTED_TARGET_LANGUAGES, config.SELECTED_TRANSLATION_ENGINES = self._saved

    def test_all_targets_in_one_batch(self) -> None:
        self.model.translator.translateCTranslate2Targets.return_value = ["Hello", "안녕", "Bonjour"]
        self.model.getTranslate = MagicMock()

        translations, flags = self.model.getInputTranslate("こんにちは", source_language="Japanese")

        self.assertEqual(translations, ["Hello", "안녕", "Bonjour"])
        self.assertEqual(flags, [True, True, True])
        kwargs = self.model.translator.translateCTranslate2Targets.call_args.kwargs
        self.assertEqual(kwargs["targets"], [
            ("English", "United States"), ("Korean", "South Korea"), ("French", "France"),
        ])
        self.model.getTranslate.assert_not_called()
        self.model._translation_executor.submit.assert_not_called()

    def test_failed_targets_fall_back_to_get_translate(self) -> None:
        self.model.translator.translateCTranslate2Targets.return_value = ["Hello", False, None]
        self.model.getTranslate = MagicMock(side_effect=[("안녕", True), ("こんにちは", True)])

        translations, flags = self.model.getInputTranslate("こんにちは", source_language="Japanese")

        self.assertEqual(translations, ["Hello", "안녕", "こんにちは"])
        self.assertEqual(flags, [True, True, True])
        called_targets = [c.args[2] for c in self.model.getTranslate.call_args_list]
        self.assertEqual(called_targets, ["Korean", "French"])


if __name__ == "__main__":
    unittest.main()
