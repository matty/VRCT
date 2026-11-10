"""文字起こしエンジンをプラガブルにするためのプロバイダ抽象化。

`transcription_transcriber.py` の `transcribeAudioQueue()` が
`match self.transcription_engine: case "Google": ... case "Whisper": ...`
という形でエンジンごとの処理をハードコードしていたのを、プロバイダ
登録ベースのディスパッチに置き換える (バックエンドレビュー Phase 3
項目17 の `TranslationProvider` レジストリと同じ考え方を文字起こし側にも
展開したもの)。

VRCT の既存パイプラインは「フレーズ確定後にまとめて1回で送信する」
バッチ型のため、プロバイダのインターフェースも同期的な `transcribe()`
のみとする (常時ストリーミング前提の非同期セッション型は採用しない)。

言語候補 (`languages`/`countries`) をまたいだ「最も確信度の高い結果を
選ぶ」ループは呼び出し元 (`transcribeAudioQueue`) に共通化し、各
プロバイダは「1つの言語候補に対して1回試行する」ことだけを担当する。
"""

import math
from typing import List, Optional, Protocol, Tuple

import numpy as np
import requests
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from speech_recognition import AudioData, Recognizer, UnknownValueError

from errors import ErrorCode
from models.transcription.transcription_deepgram import resolveDeepgramLanguageCode
from models.transcription.transcription_languages import transcription_lang
from utils import errorLogging

# setup.exe ダウンロード等の HTTP 呼び出しで使っている (10, 60) と同じ
# (connect, read) タイムアウト。実機検証の結果次第で調整する。
_HTTP_TIMEOUT = (10, 60)


class TranscriptionApiError(Exception):
    """API 系エンジンでの文字起こし失敗を、対応する `ErrorCode` 付きで表す。"""

    def __init__(self, error_code: ErrorCode, message: str = "") -> None:
        super().__init__(message or error_code.value)
        self.error_code = error_code


class TranscriptionProvider(Protocol):
    def transcribe(
        self,
        audio_data: AudioData,
        language: str,
        country: str,
        *,
        avg_logprob: float,
        no_speech_prob: float,
        no_repeat_ngram_size: int,
        force_language: bool,
    ) -> Tuple[str, float, bool]:
        """1つの言語候補に対して1回だけ試行する。

        Returns:
            (text, confidence, is_definitive) のタプル。
            - text: 認識結果 (認識できなければ "")
            - confidence: 信頼度 (呼び出し元が複数候補から最良を選ぶ)
            - is_definitive: True の場合、検出された言語が要求した言語と
              一致した等の理由でこれ以上他の候補を試す必要が無いことを
              示す (呼び出し元はこの合図でループを早期終了できる)。
        """
        ...


class GoogleProvider:
    """既存の `recognize_google` 呼び出しをそのまま包む。

    Google は候補言語ごとに別プロトコル呼び出しが必要で「検出された
    言語」という概念が無いため、`is_definitive` は常に False (呼び出し元は
    全候補を試行する、既存の挙動と同じ)。
    """

    def __init__(self, recognizer: Recognizer) -> None:
        self._recognizer = recognizer

    def transcribe(
        self,
        audio_data: AudioData,
        language: str,
        country: str,
        *,
        avg_logprob: float,
        no_speech_prob: float,
        no_repeat_ngram_size: int,
        force_language: bool,
    ) -> Tuple[str, float, bool]:
        # 2026-09-07: クリップ前後の無音パディングやクリップ分割を
        # このプロバイダ内で行う対策を試したが、実機検証の結果「クリップの
        # 前後に無音パディングを付与する」対策はエンジンを問わず
        # AudioTranscriber 側 (transcription_transcriber.py) で一律に適用する
        # 形に一本化した。Google はこれに加えて「育っていくバッファを都度
        # 再送信する」(interim_send、AudioTranscriber側) も併用しているが、
        # このプロバイダ自体は毎回渡された audio_data を1回認識するだけで
        # 良く、パディングや再送信ロジックを知る必要はない。
        try:
            # join_all_results=True: このエンドポイントは、1クリップに
            # 複数の発話区間 (無音を挟んだ複数の文) が含まれる場合、それ
            # ぞれを別々の result ブロックとして返すことがある。既定
            # (最初のブロックだけを使う) のままだと後続の発話が黙って
            # 失われる (2026-09-07、実機で確認・custom_speech_recognition
            # フォーク側で修正)。これは上記のパディング対策とは独立した
            # 別の不具合修正なので撤回せず維持する。
            text, confidence = self._recognizer.recognize_google(
                audio_data,
                language=transcription_lang[language][country]["Google"],
                with_confidence=True,
                join_all_results=True,
            )
        except UnknownValueError:
            return "", 0.0, False
        return text, confidence, False


class LocalWhisperProvider:
    """既存のローカル (faster-whisper/CTranslate2) 呼び出しをそのまま包む。"""

    def __init__(self, whisper_model) -> None:
        self._whisper_model = whisper_model

    @staticmethod
    def _toFloat32(audio_data: AudioData) -> np.ndarray:
        return np.frombuffer(
            audio_data.get_raw_data(convert_rate=16000, convert_width=2), np.int16
        ).flatten().astype(np.float32) / 32768.0

    def _start(self, raw: np.ndarray, source_language: Optional[str], avg_logprob: float, no_speech_prob: float, no_repeat_ngram_size: int):
        """`(segments, info)` を返す。segments はジェネレータで、読み出すまでデコードは走らない
        (言語の検出だけは、この呼び出しの中で済んでいる)。"""
        return self._whisper_model.transcribe(
            raw,
            beam_size=5,
            temperature=0.0,
            log_prob_threshold=avg_logprob,
            no_speech_threshold=no_speech_prob,
            language=source_language,
            word_timestamps=False,
            without_timestamps=True,
            task="transcribe",
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    @staticmethod
    def _collect(segments, avg_logprob: float, no_speech_prob: float) -> str:
        text = ""
        for s in segments:
            if s.avg_logprob < avg_logprob or s.no_speech_prob > no_speech_prob:
                continue
            text += s.text
        return text

    def transcribe(
        self,
        audio_data: AudioData,
        language: str,
        country: str,
        *,
        avg_logprob: float,
        no_speech_prob: float,
        no_repeat_ngram_size: int,
        force_language: bool,
    ) -> Tuple[str, float, bool]:
        raw = self._toFloat32(audio_data)
        source_language = transcription_lang[language][country]["Whisper"] if force_language else None
        segments, info = self._start(raw, source_language, avg_logprob, no_speech_prob, no_repeat_ngram_size)
        text = self._collect(segments, avg_logprob, no_speech_prob)

        is_definitive = force_language or transcription_lang[language][country]["Whisper"] == info.language
        return text, info.language_probability, is_definitive

    def transcribe_candidates(
        self,
        audio_data: AudioData,
        candidates: List[Tuple[str, str]],
        *,
        avg_logprob: float,
        no_speech_prob: float,
        no_repeat_ngram_size: int,
    ) -> Tuple[str, float, int]:
        """候補言語 (language, country) が複数あるときに 1 回で文字起こしする。

        Returns: (text, confidence, 採用した候補の添字)。

        言語を指定しない呼び出しは、どの候補に対して呼んでも同じ結果を返す
        (temperature 0 なので決定的)。候補ごとに `transcribe(force_language=False)` を
        呼ぶ従来の方法には問題が 2 つあった。

        - 検出された言語が最初の候補でないと、一致する候補に当たるまで同じ音声を
          デコードし直していた。候補に無い言語 (自分と同じ言語で話す人など) なら
          候補の数だけ。候補 3 つなら文字起こしに 3 倍かかる。
        - どの呼び出しも同じ confidence を返すので、呼び出し元の「confidence が
          より高い結果を採用」では常に最初の候補が残り、検出された言語が 2 番目
          以降の候補でも、結果の言語は最初の候補になっていた (翻訳の原文言語を
          取り違える)。

        検出された言語が候補にあればその候補を、無ければ従来どおり最初の候補を返す。

        「候補に無い言語が検出されたら、候補の中で最も確率の高い言語でデコードし
        直す」方法も評価したが採用しなかった (改善と悪化が同程度。
        docs/transcription_language_candidates_2026-11-10.md)。
        """
        raw = self._toFloat32(audio_data)
        codes = [transcription_lang[language][country]["Whisper"] for language, country in candidates]
        segments, info = self._start(raw, None, avg_logprob, no_speech_prob, no_repeat_ngram_size)
        text = self._collect(segments, avg_logprob, no_speech_prob)
        index = codes.index(info.language) if info.language in codes else 0
        return text, info.language_probability, index


# Whisper の言語名のうち、VRCT の表示名 (transcription_lang のキー) と違うもの。
# OpenAI 公式の verbose_json は検出した言語を名前で返す ("chinese" など)。
_WHISPER_LANGUAGE_NAME_TO_CODE = {
    "chinese": "zh",
    "mandarin": "zh",
    "cantonese": "yue",
    "tagalog": "tl",
    "myanmar": "my",
}


def _map_openai_exception(exc: Exception) -> ErrorCode:
    if isinstance(exc, AuthenticationError):
        return ErrorCode.TRANSCRIPTION_API_AUTH_FAILED
    if isinstance(exc, RateLimitError):
        return ErrorCode.TRANSCRIPTION_API_RATE_LIMITED
    if isinstance(exc, APITimeoutError):
        return ErrorCode.TRANSCRIPTION_API_TIMEOUT
    if isinstance(exc, (APIStatusError, APIConnectionError)):
        return ErrorCode.TRANSCRIPTION_API_SERVER_ERROR
    return ErrorCode.TRANSCRIPTION_API_SERVER_ERROR


class OpenAICompatibleTranscriptionProvider:
    """OpenAI互換の音声書き起こしREST API (`/v1/audio/transcriptions`) 向け
    プロバイダ。公式 `openai` パッケージ (既存依存) の
    `client.audio.transcriptions.create(...)` を使う。

    Groq/OpenAI公式/issue #100 のカスタムローカルサーバーを、
    base_url/api_key/model の差し替えだけで1つの実装でカバーする
    (ユーザーから見えるエンジン選択肢としては、翻訳側の
    `OpenAI_API`/`Groq_API`/`OpenAI_Compatible` が別々の選択肢であるのと
    同じ構造で、`Groq_Whisper`/`OpenAI_Whisper`/`Custom_Whisper` として
    それぞれ専用のインスタンスを持つ)。

    音声フォーマットは各API先の仕様に準拠する方針のため、現時点では
    最も広くサポートされている WAV (`AudioData.get_wav_data()`) を使う。
    """

    def __init__(self, api_key: str, base_url: str, model: str, engine_name: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.engine_name = engine_name
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=_HTTP_TIMEOUT)

    def _request(
        self,
        audio_data: AudioData,
        source_language: Optional[str],
        avg_logprob: float,
        no_speech_prob: float,
    ) -> Tuple[str, float, Optional[str]]:
        """API を 1 回呼ぶ。(text, confidence, API が検出した言語) を返す。"""
        wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)

        try:
            response = self._client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes, "audio/wav"),
                model=self.model,
                language=source_language,
                response_format="verbose_json",
                temperature=0.0,
            )
        except Exception as exc:
            error_code = _map_openai_exception(exc)
            errorLogging()
            raise TranscriptionApiError(error_code) from exc

        detected_language = getattr(response, "language", None)
        segments = getattr(response, "segments", None) or []
        accepted_logprobs: List[float] = []
        text = ""
        if segments:
            for s in segments:
                if s.avg_logprob < avg_logprob or s.no_speech_prob > no_speech_prob:
                    continue
                text += s.text
                accepted_logprobs.append(s.avg_logprob)
        else:
            # verbose_json 非対応のサーバー (カスタムサーバー等) は
            # segments を返さないことがある。その場合はセグメント単位の
            # フィルタリングができないため、テキストをそのまま採用する。
            text = getattr(response, "text", "") or ""

        if not text:
            return "", 0.0, detected_language

        # avg_logprob (対数確率、概ね負の値) を 0〜1 の疑似的な信頼度に変換する。
        # ローカル Whisper の info.language_probability に相当するものが
        # verbose_json には無いため、代替の指標として使う。
        confidence = math.exp(sum(accepted_logprobs) / len(accepted_logprobs)) if accepted_logprobs else 0.5
        return text, confidence, detected_language

    def _isDetectedLanguage(self, detected_language: Optional[str], language: str, country: str) -> bool:
        """API が返した言語が、この候補 (language, country) を指しているか。

        verbose_json の `language` は実装によって形式が違う。OpenAI 公式は
        "japanese" のような言語名、faster-whisper 系のサーバーは "ja" のような
        コードを返す。コードだけと比べていたため、OpenAI 公式では一度も一致せず、
        候補の数だけ同じ音声を送り直していた。
        """
        if not isinstance(detected_language, str) or not detected_language:
            return False
        detected = detected_language.strip().lower()
        code = transcription_lang[language][country][self.engine_name]
        return detected in (code.lower(), language.lower()) or _WHISPER_LANGUAGE_NAME_TO_CODE.get(detected) == code

    def transcribe(
        self,
        audio_data: AudioData,
        language: str,
        country: str,
        *,
        avg_logprob: float,
        no_speech_prob: float,
        no_repeat_ngram_size: int,
        force_language: bool,
    ) -> Tuple[str, float, bool]:
        source_language = transcription_lang[language][country][self.engine_name] if force_language else None
        text, confidence, detected_language = self._request(audio_data, source_language, avg_logprob, no_speech_prob)
        if not text:
            return "", 0.0, False
        is_definitive = force_language or self._isDetectedLanguage(detected_language, language, country)
        return text, confidence, is_definitive

    def transcribe_candidates(
        self,
        audio_data: AudioData,
        candidates: List[Tuple[str, str]],
        *,
        avg_logprob: float,
        no_speech_prob: float,
        no_repeat_ngram_size: int,
    ) -> Tuple[str, float, int]:
        """候補言語が複数あるときに、API を 1 回だけ呼んで文字起こしする。

        Returns: (text, confidence, 採用した候補の添字)。

        言語を指定しない呼び出しは、どの候補に対して呼んでも同じ結果を返す。
        従来は検出された言語が候補に一致するまで候補の数だけ呼び直しており、
        同じ音声に何度も課金され、その分だけ遅くなっていた。1 回の結果を、
        検出された言語に一致する候補 (無ければ最初の候補) の結果として返す。
        """
        text, confidence, detected_language = self._request(audio_data, None, avg_logprob, no_speech_prob)
        for index, (language, country) in enumerate(candidates):
            if self._isDetectedLanguage(detected_language, language, country):
                return text, confidence, index
        return text, confidence, 0


_DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"


def _map_deepgram_status(status_code: int) -> ErrorCode:
    if status_code in (401, 403):
        return ErrorCode.TRANSCRIPTION_API_AUTH_FAILED
    if status_code == 429:
        return ErrorCode.TRANSCRIPTION_API_RATE_LIMITED
    return ErrorCode.TRANSCRIPTION_API_SERVER_ERROR


class DeepgramProvider:
    """Deepgram の録音済み(バッチ) REST API (`/v1/listen`) 向けプロバイダ。

    OpenAI互換ではないため `openai` パッケージは使わず、既存依存の
    `requests` で直接叩く。

    言語コードについて: 候補言語が1つに確定している場合
    (`force_language=True`)、`resolveDeepgramLanguageCode()` で
    このモデルが実際に対応言語として申告しているコード (Google列との
    完全一致、または Whisper列のベースコード一致) を解決できれば、
    それを `language=` として明示的に渡す。解決できない場合
    (対応コードが不明、または候補言語が複数で1つに絞れない場合) は
    `detect_language=true` (自動検出) にフォールバックする。
    いずれの経路でも1回のAPI呼び出しで完結するため
    (Deepgram自身が1呼び出しで多言語を判定できる、Whisper系のように
    候補言語ごとに複数回呼ぶ必要が無い)、`is_definitive` は常に True を
    返し、呼び出し元のループを1回で打ち切らせる。

    信頼度についても、avg_logprob/no_speech_prob に相当するセグメント
    単位の指標をDeepgramは返さないため、トップレベルの `confidence`
    (0〜1) をそのまま使う (セグメント単位のフィルタリングは行わない)。
    """

    def __init__(self, api_key: str, model: str, model_languages: Optional[List[str]] = None) -> None:
        self.api_key = api_key
        self.model = model
        self.model_languages = model_languages or []

    def transcribe(
        self,
        audio_data: AudioData,
        language: str,
        country: str,
        *,
        avg_logprob: float,
        no_speech_prob: float,
        no_repeat_ngram_size: int,
        force_language: bool,
    ) -> Tuple[str, float, bool]:
        wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)

        params = {"model": self.model}
        resolved_code = resolveDeepgramLanguageCode(language, country, self.model_languages) if force_language else None
        if resolved_code:
            params["language"] = resolved_code
        else:
            params["detect_language"] = "true"

        try:
            response = requests.post(
                _DEEPGRAM_LISTEN_URL,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "audio/wav",
                },
                params=params,
                data=wav_bytes,
                timeout=_HTTP_TIMEOUT,
            )
        except requests.exceptions.Timeout as exc:
            errorLogging()
            raise TranscriptionApiError(ErrorCode.TRANSCRIPTION_API_TIMEOUT) from exc
        except requests.exceptions.RequestException as exc:
            errorLogging()
            raise TranscriptionApiError(ErrorCode.TRANSCRIPTION_API_SERVER_ERROR) from exc

        if response.status_code != 200:
            errorLogging()
            raise TranscriptionApiError(_map_deepgram_status(response.status_code))

        payload = response.json()
        try:
            channel = payload["results"]["channels"][0]
            alternative = channel["alternatives"][0]
        except (KeyError, IndexError):
            return "", 0.0, False

        text = alternative.get("transcript", "") or ""
        if not text:
            return "", 0.0, False

        confidence = float(alternative.get("confidence", 0.0) or 0.0)
        # 明示コード・自動検出のいずれでも1回の呼び出しで完結するため、
        # この1回の結果が最終結果。呼び出し元 (transcribeAudioQueue) には
        # 他の候補言語を試させない。
        return text, confidence, True
