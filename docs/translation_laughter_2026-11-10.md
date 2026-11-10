# ローカル翻訳モデルでの笑いの表現 (2026-11-10)

`docs/translation_local_models_2026-11-08.md` で今後の課題にしていた
「www」「草」の正規化。対象は CTranslate2 (NLLB-200 / M2M100) だけ。

## 1. 問題

NLLB / M2M100 は書き言葉の文で学習されており、チャットの笑いの表現を知らない。
そのまま残すのではなく、存在しない文を作る。汚い言葉になることもある。

NLLB-200-distilled-600M int8 での実測 (変更前 → 変更後):

| 原文 | 変更前 | 変更後 |
|---|---|---|
| www | I'm not sure. | lol |
| 草 | The grass. | lol |
| 大草原 | The Great Plains. | lol |
| ワロタ | I'm going to warrant it. | lol |
| それなwww | That's the Internet. | That's it lol |
| それはやばいwww | It's a f\*\*\*ing world. | That's a b\*\*\*h lol (※) |
| まじかwww それはやばいwww | It's not a website. It's a website. | I don't know lol That's a b\*\*\*h lol (※) |
| 今日めっちゃ楽しかった笑 | I had a really good laugh today. | I had a really good time today lol |
| それは草 | It's grass. | That's what it is lol |
| ㅋㅋㅋ (ko→en) | I did. | lol |
| 진짜 웃기다 ㅋㅋㅋㅋ (ko→en) | That's really funny. | It's really funny lol |
| 哈哈哈 (zh→en) | What is it? | lol |
| 笑死我了 (zh→en) | You're laughing at me. | lol |
| jajaja (es→en) | Oh, my God. | lol |
| qué gracioso jajaja (es→en) | What a funny s\*\*t. | How funny lol |
| kkkkk (pt→en) | What are you doing? | lol |
| 555 (th→en) | 555 of them. | lol |
| хахаха (ru→en) | What's the matter? | lol |
| lol (en→ja) | ロル | 笑 |
| lmao (en→ja) | イマオ | 笑 |
| that's so funny lol (en→ja) | 面白かったよ | 面白かった笑 |
| no way lmao (en→ja) | まったくないわ | 絶対にない笑 |
| lol (en→ko) | 로리 | ㅋㅋㅋ |

(※) 「やばい」を単独で訳した結果。笑いの表現とは別の、モデル自体の俗語の弱さ。

笑いの表現を含まない文は変更前と同じ訳になる
(「庭の草を刈った」→ I cut the grass in the garden.、「wwwにアクセスする」→ You can access the website.)。

## 2. 変更

`src-python/models/translation/translation_laughter.py`

- 原文の言語ごとの正規表現で笑いの表現を見つけ、原文を「文」と「笑い」に分ける。
- 文だけをモデルに渡す (既存の文分割・一括翻訳にそのまま乗る)。笑いは翻訳先の言語の
  書き方で元の位置に戻す。笑いだけのメッセージはモデルを呼ばない。
- 笑いの直前の句点は取る (「That's crazy. lol」ではなく「That's crazy lol」)。
- `Translator.translate()` と `translateCTranslate2Targets()` の CTranslate2 経路だけで使う。
  言語名を渡さずに `translateCTranslate2Batch()` を直接呼んだ場合は従来どおり。

| 原文の言語 | 笑いとして扱うもの |
|---|---|
| 日本語 | w / ｗ の連続、草 (単独・助詞の後)、大草原、草生える、笑、(笑)、ワロタ、ははは |
| 韓国語 | ㅋ / ㅎ の連続 |
| 中国語 | 哈哈 / 呵呵 / 嘿嘿 / 嘻嘻、笑死(我了)、hhh、2333 |
| 英語 | lol、lmao、rofl など (節の先頭か末尾のときだけ) |
| スペイン語 / ポルトガル語 / タイ語 / ロシア語 / インドネシア語 / フランス語 | jajaja / kkkk・rsrs / 555 / хаха / wkwk / mdr |
| 全言語 | haha、hehe など |

| 翻訳先 | 書き方 |
|---|---|
| 英語 | lol |
| 日本語 | 笑 (空白なしで文に続ける) |
| 韓国語 | ㅋㅋㅋ |
| 中国語 | 哈哈哈 (空白なし) |
| スペイン語 / ポルトガル語 / タイ語 / ロシア語 / インドネシア語 / フランス語 | jajaja / kkkk / 555 / ахаха / wkwk / mdr |
| その他 | haha |

### 笑いとして扱わないもの (普通の語と区別する)

- 「庭の草を刈った」「草を食べる」: 助詞「の」「を」に挟まれた草。
- 「wwwにアクセスする」「www.example.com」: 後ろに文字やドメインが続く www。
- 「爆笑した」「笑う」「彼は笑った」: 語の一部の 笑。
- 「5wの電球」「wifi」: 後ろに文字が続く w。
- "I play lol every day": 節の途中の lol (ゲーム名)。"that's funny lol" や "lol what" は笑い。
- 「ราคา 555 บาท」「我有2333个」: 文の途中の数字。

## 3. 限界

- 対象は CTranslate2 だけ。Google / Bing / Papago / DeepL / LLM は測っていないので
  触っていない (チャットの俗語を知っていることが多い)。
- 笑いの強さ (www と wwwwwwww) は区別しない。
- 原文の言語の表現だけを見る (英語を選んでいるときに打った「ㅋㅋㅋ」はそのままモデルに渡る)。
- 文の途中に空白なしで入った短い笑い (「いやwそれは」) は拾わない (w が 3 つ以上なら拾う)。
- 笑い以外の俗語 (やばい、まじか) の誤訳は直らない。
