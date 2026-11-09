"""近くの話者の声だけを残し、遠くの話し声を落とす (レベルによる判定)。

VRChat では、目的の声と遠くの声はレベルで分けられることが多い:

- マイク: 自分の口はヘッドセットのマイクから数 cm。同じ部屋の人の声や
  スピーカーから漏れたゲーム音声は 20〜30 dB 小さい。
- スピーカー (ループバック): VRChat はプレイヤーの声を距離に応じて
  小さくするので、近くのプレイヤーほど大きい。

そこで「近くの話者のレベル」(`reference_db`) をクリップをまたいで追跡し、
それより `margin_db` 以上小さい区間を遠くの声とみなして落とす:

1. クリップ全体が遠くの声だけなら `None` を返す (呼び出し側は文字起こし
   自体を行わない = 他人の雑談を翻訳して送らない)。
2. 近くの声を含むクリップでは、遠くの声だけの区間 (近くの話者の前後や
   間 (ま)) を減衰させる。近くの話者の子音 (母音より 10〜20 dB 小さい) を
   削らないよう、近くの声と判定したフレームの前後 `hold_ms` は残す。

基準は「直近 `window_sec` 秒に近くの声と判定したクリップのレベルの中央値」
(最初の `min_history` クリップは落とさずに基準づくりに使う)。
最大値を使うと、一度笑ったり叫んだりしただけで基準が跳ね上がり、普段の声まで
遠くの声として削ってしまう。遠くの声だけのクリップは履歴に入れないので、
周りの雑談が続いても基準は下がらない。履歴が空 (しばらく誰も近くで話して
いない) なら、そのクリップ自身のレベルを基準にする。

限界 (評価: docs/transcription_noise_reduction_2026-11-08.md):

- 近くの声と遠くの声が同時に鳴っている区間は分離できない (時間方向に
  切ることしかできない)。その区間は近くの声が支配的なので Whisper は概ね
  近くの声を書き起こす。
- レベルだけでは「マイクの音量が小さい近くの人」と「遠くの人」を区別
  できない。スピーカー側で、近くにいてもマイクが `margin_db` 以上小さい
  プレイヤーの声は落ちる。
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple

import numpy as np

_FRAME_SECONDS = 0.032


@dataclass(frozen=True)
class NearVoiceSettings:
    # 基準 (近くの話者のレベル) からこれ以上小さい区間を遠くの声とみなす。
    margin_db: float = 12.0
    # クリップ内でフレームを削る基準。None なら margin_db と同じ。
    # クリップ全体を落とすかどうかとは別に、近くの声を含むクリップの中で
    # どこまで削るかを決める (大きいほど控えめ)。
    frame_margin_db: Optional[float] = None
    # 近くの声と判定したフレームの前後に残す時間 (子音・語尾を削らないため)。
    hold_ms: float = 400.0
    # 遠くの声の区間の減衰量。
    attenuation_db: float = 40.0
    # 基準を決めるのに使う、近くの声のクリップの履歴の長さ。
    window_sec: float = 120.0
    # 履歴がこれ未満の間はクリップを落とさない (基準を作っている途中)。
    # 最初のクリップがたまたま大きい声 (笑い声など) だと、それだけで基準が
    # 決まり、普段の声が 2 分間落ち続けるのを防ぐ。
    min_history: int = 3
    # クリップのレベル = 大きい方からこの割合のフレームの平均 (dB)。
    clip_level_top_fraction: float = 0.1
    # これより小さいクリップのレベルは基準に使わない (無音・ノイズ)。
    min_level_dbfs: float = -55.0


def _frame_db(samples: np.ndarray, frame: int) -> np.ndarray:
    count = samples.size // frame
    if count == 0:
        return np.zeros(0)
    frames = samples[: count * frame].reshape(count, frame).astype(np.float64)
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


class NearVoiceGate:
    """クリップをまたいで近くの話者のレベルを追跡する状態付きのゲート。

    mic/speaker の `AudioTranscriber` ごとに 1 つ持つ。スレッドセーフではない
    (1 つの文字起こしスレッドからだけ呼ぶ)。
    """

    def __init__(
        self,
        settings: NearVoiceSettings = NearVoiceSettings(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._clock = clock
        self._history: Deque[Tuple[float, float]] = deque()
        # 直近の判定に使った基準 (ログ用)。
        self.reference_db: Optional[float] = None

    def reset(self) -> None:
        self._history.clear()
        self.reference_db = None

    def process(self, samples: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """[-1, 1] のモノラル float32 を受け取り、遠くの声を落とした音声を返す。

        クリップ全体が遠くの声だけなら None。
        """
        s = self.settings
        frame = max(1, int(round(sample_rate * _FRAME_SECONDS)))
        levels = _frame_db(samples, frame)
        # 前後のデジタル無音 (VAD のパディング) はレベル計算に含めない。
        active = levels > -100
        if not np.any(active):
            return samples
        active_levels = levels[active]
        top = max(1, int(np.ceil(active_levels.size * s.clip_level_top_fraction)))
        clip_level = float(np.mean(np.sort(active_levels)[-top:]))
        if clip_level < s.min_level_dbfs:
            return samples

        now = self._clock()
        while self._history and now - self._history[0][0] > s.window_sec:
            self._history.popleft()
        if len(self._history) < s.min_history:
            # 基準を作っている途中: 落とさずに履歴に入れ、このクリップも含めた中央値で判定する。
            self._history.append((now, clip_level))
            reference = float(np.median([level for _, level in self._history]))
            self.reference_db = reference
            threshold = reference - s.margin_db
        else:
            reference = float(np.median([level for _, level in self._history]))
            self.reference_db = reference
            threshold = reference - s.margin_db
            if clip_level < threshold:
                return None
            self._history.append((now, clip_level))

        frame_margin = s.margin_db if s.frame_margin_db is None else s.frame_margin_db
        near = levels >= reference - frame_margin
        hold = int(round(s.hold_ms / 1000 / _FRAME_SECONDS))
        if hold > 0 and near.size:
            # 前後 hold フレームに広げる (近くの声のフレームの近傍は残す)。
            padded = np.concatenate([np.zeros(hold, bool), near, np.zeros(hold, bool)])
            csum = np.concatenate([[0], np.cumsum(padded)])
            window = 2 * hold + 1
            near = (csum[window:] - csum[:-window]) > 0
        floor = 10 ** (-s.attenuation_db / 20)
        frame_gain = np.where(near, 1.0, floor)
        # フレーム境界でゲインが跳ばないよう、サンプル単位で線形補間する。
        centers = (np.arange(frame_gain.size) + 0.5) * frame
        gain = np.interp(np.arange(samples.size), centers, frame_gain)
        return (samples * gain).astype(np.float32)
