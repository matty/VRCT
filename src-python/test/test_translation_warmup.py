"""ロード/モデル選択直後の待ち時間を減らす処理のテスト。

- CTranslate2: ロード直後にバックグラウンドで 1 回訳しておく
  (1 件目だけ遅い: NLLB-600M int8 / CPU の実測で約 680ms → 以降約 360ms)
- Ollama: モデルを選んだらバックグラウンドで事前ロードする
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

import models.translation.translation_ollama as ollama_module
import models.translation.translation_translator as tt
from models.translation.translation_ollama import OllamaClient
from models.translation.translation_translator import Translator


class _Result:
    def __init__(self, tokens):
        self.hypotheses = [tokens]


class _FakeCT2Translator:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def translate_batch(self, sources, target_prefix, **kwargs):
        self.calls.append(target_prefix)
        return [_Result(prefix + ["x"]) for prefix in target_prefix]


class _FakeTokenizer:
    lang_code_to_token = {"ja": "__ja__"}
    src_lang = None

    def encode(self, message):
        return list(message)

    def convert_ids_to_tokens(self, ids):
        return ids

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def decode(self, ids):
        return "".join(ids)


class _SyncThread:
    """Thread の代わりに、start() でその場で実行する。"""

    def __init__(self, target, name=None, daemon=None):
        self.target = target
        self.name = name

    def start(self):
        self.target()


class CTranslate2WarmupTests(unittest.TestCase):
    def _load(self, model_type):
        translator = Translator()
        fake_ct2 = MagicMock()
        fake_ct2.Translator.side_effect = _FakeCT2Translator
        fake_transformers = MagicMock()
        fake_transformers.AutoTokenizer.from_pretrained.return_value = _FakeTokenizer()
        with patch.object(tt, "ctranslate2", fake_ct2), \
             patch.object(tt, "transformers", fake_transformers), \
             patch.object(tt, "Thread", _SyncThread), \
             patch.object(tt, "getBestComputeType", return_value="int8"):
            translator.changeCTranslate2Model(".", model_type)
        return translator

    def test_nllb_is_warmed_up_after_load(self):
        translator = self._load("nllb-200-distilled-600M-ct2-int8")
        self.assertEqual(translator.ctranslate2_translator.calls, [[["jpn_Jpan"]]])
        self.assertEqual(len(translator._cache), 0, "ウォームアップの訳はキャッシュに入れない")

    def test_m2m100_is_warmed_up_with_its_language_token(self):
        translator = self._load("m2m100_418M-ct2-int8")
        self.assertEqual(translator.ctranslate2_translator.calls, [[["__ja__"]]])

    def test_warmup_runs_in_background_and_waits_for_the_lock(self):
        translator = Translator()
        translator.is_loaded_ctranslate2_model = True
        translator.ctranslate2_tokenizer = _FakeTokenizer()
        translator.ctranslate2_translator = _FakeCT2Translator()
        with translator._ctranslate2_lock:
            translator._startCTranslate2Warmup("nllb-200-distilled-600M-ct2-int8")
            # ロード側がロックを持っている間はまだ走らない
            self.assertEqual(translator.ctranslate2_translator.calls, [])
        for t in threading.enumerate():
            if t.name == "ct2-warmup":
                t.join(timeout=5)
        self.assertEqual(translator.ctranslate2_translator.calls, [[["jpn_Jpan"]]])


class OllamaPreloadTests(unittest.TestCase):
    def test_selecting_a_model_preloads_it(self):
        client = OllamaClient()
        client.model = "hy-mt1.5-1.8b"
        with patch.object(ollama_module.threading, "Thread", _SyncThread), \
             patch.object(ollama_module.requests, "post") as post:
            client.updateClient()
        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/generate")
        # prompt なし = ロードだけ
        self.assertEqual(kwargs["json"], {"model": "hy-mt1.5-1.8b", "keep_alive": "30m"})

    def test_preload_failure_is_ignored(self):
        client = OllamaClient()
        client.model = "qwen3:8b"
        with patch.object(ollama_module.threading, "Thread", _SyncThread), \
             patch.object(ollama_module.requests, "post", side_effect=ConnectionError):
            client.updateClient()  # 例外にならない

    def test_no_model_no_preload(self):
        client = OllamaClient()
        with patch.object(ollama_module.requests, "post") as post:
            client._preloadModel()
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
