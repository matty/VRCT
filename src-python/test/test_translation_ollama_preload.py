"""Ollama: モデルを選んだらバックグラウンドで事前ロードすることのテスト。"""

import unittest
from unittest.mock import patch

import models.translation.translation_ollama as ollama_module
from models.translation.translation_ollama import OllamaClient


class _SyncThread:
    """Thread の代わりに、start() でその場で実行する。"""

    def __init__(self, target, name=None, daemon=None):
        self.target = target
        self.name = name

    def start(self):
        self.target()


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
