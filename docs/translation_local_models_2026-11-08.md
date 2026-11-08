# ローカル翻訳モデルの改善と調査 (2026-11-08)

## 1. CTranslate2 (既定エンジン、NLLB-200 / M2M100) の改善

| 変更 | 理由 |
|---|---|
| 文に分割して翻訳し、つなぎ直す | NLLB/M2M100 は文単位で学習されたモデル。複数文をまとめて渡すと後半の文を落とす |
| 短すぎる文 (特殊トークン込み 8 未満) は前後とまとめる | 「はじめまして！」を単独で訳すと文脈が無く "You're welcome!" になった |
| 複数ターゲット言語を `translate_batch` 1 回にまとめる | `_ctranslate2_lock` で直列化されるため、スレッドプールで並列に投げても速くならなかった |
| 出力長の上限 = 入力トークン数 × 3 + 16 (最大 256) | 幻覚で同じ句を繰り返し続けたときに既定の 256 トークンまで生成させない (保険) |
| 決定的なエンジン (Google/Bing/Papago/CTranslate2) の結果を LRU (512 件) でキャッシュ | 短い定型句が繰り返し出る。LLM と DeepL は会話の文脈で訳が変わるので対象外 |

### 実測 (NLLB-200-distilled-600M int8, CPU 4 コア, Linux)

旧実装 (文をまとめて 1 回、ターゲットごとに逐次) と新実装の比較:

- 3 ターゲット言語 (日・韓・中) の翻訳: 中央値 **1482–1652 ms → 684–702 ms (約 2.1–2.4 倍)**
- キャッシュヒット: 0.02 ms
- 文の欠落の修正例 (ja→en):
  - 原文: はじめまして！日本から来ました。英語はあまり上手じゃないけど、よろしくお願いします。このワールドはとても綺麗ですね。
  - 旧: Hello! I'm from Japan. I'm not very good at English, but I'd like to welcome you. ← **最後の文が消えている**
  - 新: Hello, I'm from Japan. I'm not very good at English, but I'd like to welcome you. This world is so beautiful.
- en→ja "See you tomorrow, okay?": 旧「明日見ます」→ 新「明日会えるよ」

繰り返しを起こしやすい入力 (hahaha…, あああ…, ok ok ok…) では、今回の試行で暴走は再現せず、出力も旧実装と同一だった。上限は最悪時の保険として入れている。

### 検討して入れなかったもの (CTranslate2)

**ロード直後のウォームアップ**: 一度入れたが取り消した。1 件目だけ遅い (約 680ms vs 360ms) という
計測は、ディスクから冷えた状態でロードした直後の 1 回だけのものだった。CPU が空いた状態で
プロセスを分けて測り直すと、ロード直後の 1 件目も 2 件目以降と変わらなかった
(376 / 350ms、329 / 317ms)。

**`CT2_USE_EXPERIMENTAL_PACKED_GEMM=1`** (重み行列をロード時に MKL 用に詰め直す):

| NLLB-600M int8, Intel Xeon 4 コア | 既定 | packed |
|---|---|---|
| 1 文あたりの中央値 (3 回) | 398–416 ms | 285–328 ms (約 25% 速い) |
| モデルのメモリ (RSS) | +524 MB | +1109 MB |
| 訳文 | 同一 | 同一 |

メモリがほぼ倍になる (NLLB-3.3B なら数 GB 増える見込み) ため既定では有効にしない。
さらに次の制約がある。

- CTranslate2 はこの値をプロセス内で最初にモデルを読むときに 1 回だけ読む。後から設定しても
  効かず、同じ CTranslate2 で動く Whisper (文字起こし) にも適用される (Whisper への影響は未計測)。
- MKL を使う Intel CPU でだけ効く。実験的な機能。

メモリに余裕があり速度を優先したい場合は、VRCT を起動する前に環境変数
`CT2_USE_EXPERIMENTAL_PACKED_GEMM=1` を設定すれば有効になる。

## 2. LLM エンジン (Ollama / LM Studio / クラウド) の改善

- 既定プロンプトから対応言語の一覧 (毎回数百トークン) を削除。ローカル LLM では prefill が短くなる分だけ速くなる
- 「入力は常に翻訳対象で、質問されても答えない」を明記 (翻訳せずに返答してしまう誤りへの対策)
- 会話履歴は `<conversation>` で囲み「参照専用・翻訳しない」と明記。発話者は mic/chat → Me、speaker/ocr → Other
- 出力から `<think>…</think>`、全体を囲む引用符、"Translation:" を除去
- 空の訳は失敗として扱い、CTranslate2 へフォールバックする
- Ollama の `keep_alive` を 30 分に延長 (既定の 5 分では会話が途切れるたびにモデルを再ロードしていた)
- DeepL に直近の会話を `context` として渡す (課金対象外)

## 3. 翻訳専用ローカルモデルへの対応 (調査結果)

2025〜2026 年の翻訳専用モデルは、Ollama / LM Studio で動かせる LLM が中心。どれも
**学習時の固定プロンプトから外れると精度が落ちる**ため、モデル名で判別して公式の形式で
問い合わせるようにした (`translation_llm_base.detectModelProfile`)。

| モデル | 規模 | 言語 | 対応内容 | 出典 |
|---|---|---|---|---|
| Tencent HY-MT1.5 | 1.8B / 7B | 33 + 方言 5 | 公式テンプレート (中国語が絡む対は中国語の指示文)、文脈付きテンプレート、推奨サンプリング (temperature 0.7, top_p 0.6, top_k 20, repetition_penalty 1.05) | https://huggingface.co/tencent/HY-MT1.5-1.8B |
| Tencent Hunyuan-MT-7B | 7B | 33 | 同上 (文脈テンプレートなし)。WMT25 で 31 中 30 の言語対で 1 位 (Tencent 発表) | https://huggingface.co/tencent/Hunyuan-MT-7B |
| Google TranslateGemma | 4B / 12B / 27B | 55 | 公式テンプレート (言語名 + コード、本文の前に空行 2 つ)。文脈欄が無いので履歴は渡さない | https://ollama.com/library/translategemma |

おすすめ: VRChat と同時に動かすなら **HY-MT1.5-1.8B** (GGUF で 1GB 前後、技術報告の平均応答 0.18 秒)。
GPU に余裕があれば TranslateGemma 12B / Hunyuan-MT-7B。

### 今回は入れなかったもの

- **MADLAD-400 (Google, Apache-2.0, 400+ 言語)**: CTranslate2 で動き、ライセンスは NLLB (CC-BY-NC) より扱いやすい。
  ただし最小が 3B で NLLB-600M より大きく遅いうえ、主要言語の品質は NLLB-1.3B/3.3B と大差ないとされる。
  UI の重み選択・ダウンロード・言語表 (`<2xx>` 形式) の追加が必要なので、評価してから別途検討する。
- Seed-X-PPO-7B: Hunyuan-MT の比較表で下回っているため見送り。
- 日本語の「www」「草」は NLLB が "I'm not sure." "It's grass." と訳す。笑いの表現の正規化は今後の課題。
