import io
import os
import tempfile
import modal

app = modal.App("transcription-asr-v1")


def download_asr_model():
    from nemo.collections.asr.models import ASRModel

    print("Downloading ASR model...")
    ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt_ctc-0.6b-ja")
    print("ASR model cached.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "Cython",
        "nemo_toolkit[asr]",
        "pydub",
    )
    .run_function(download_asr_model)
)


@app.cls(
    image=image,
    gpu="T4",
    timeout=3600,
    scaledown_window=2,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class ASRService:
    @modal.enter(snap=True)
    def load_model(self):
        from nemo.collections.asr.models import ASRModel
        from nemo.collections.asr.parts.utils.asr_confidence_utils import (
            ConfidenceConfig,
            ConfidenceMethodConfig,
        )
        from omegaconf import OmegaConf, open_dict

        self.asr_model = ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt_ctc-0.6b-ja"
        )
        self.asr_model.to("cuda")
        self.asr_model.eval()

        # Confidence を有効化
        confidence_cfg = ConfidenceConfig(
            preserve_frame_confidence=False,
            preserve_token_confidence=True,   # token(char)レベル
            preserve_word_confidence=True,    # wordレベル
            exclude_blank=True,
            aggregation="mean",
            tdt_include_duration=False,
            method_cfg=ConfidenceMethodConfig(
                name="max_prob",  # or "entropy"
            ),
        )

        decoding_cfg = self.asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.confidence_cfg = OmegaConf.structured(confidence_cfg)
            # タイムスタンプも同時に有効化
            decoding_cfg.preserve_alignments = True
            decoding_cfg.compute_timestamps = True

        self.asr_model.change_decoding_strategy(decoding_cfg)

    def _safe_get(self, obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _normalize_timestamp_items(self, items, segment_start_sec, text_key):
        normalized = []
        for x in items or []:
            token_text = self._safe_get(x, text_key, None)
            start = self._safe_get(x, "start", None)
            end = self._safe_get(x, "end", None)
            confidence = self._safe_get(x, "confidence", None)

            if token_text is None or start is None or end is None:
                continue

            normalized.append(
                {
                    text_key: token_text,
                    "start": float(start) + segment_start_sec,
                    "end": float(end) + segment_start_sec,
                    "confidence": float(confidence) if confidence is not None else None,
                }
            )
        return normalized

    @modal.method()
    def transcribe_segment(
        self,
        segment_data: bytes,
        segment_start_sec: float,
        segment_end_sec: float,
    ):
        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(segment_data)
                temp_audio_path = temp_audio.name

            transcriptions = self.asr_model.transcribe(
                [temp_audio_path],
                return_hypotheses=True,
                timestamps=True,
            )

            first_result = transcriptions[0]

            raw_text = self._safe_get(first_result, "text", str(first_result))
            timestamp_obj = self._safe_get(first_result, "timestamp", {}) or {}

            chars = self._safe_get(timestamp_obj, "char", []) or []
            words = self._safe_get(timestamp_obj, "word", []) or []
            segments = self._safe_get(timestamp_obj, "segment", []) or []

            # hyp.token_confidence / hyp.word_confidence は timestamp とは別リストで来る
            # timestamp['char'][i] には confidence キーがないので、インデックスで突き合わせる
            raw_token_conf = self._safe_get(first_result, "token_confidence", None) or []
            raw_word_conf  = self._safe_get(first_result, "word_confidence",  None) or []

            def _to_float(v):
                """tensor / float / None を float | None に変換"""
                if v is None:
                    return None
                try:
                    return float(v)
                except Exception:
                    return None

            token_conf_list = [_to_float(v) for v in raw_token_conf]
            word_conf_list  = [_to_float(v) for v in raw_word_conf]

            adjusted_chars = []
            for i, c in enumerate(chars):
                ch    = self._safe_get(c, "char",  None)
                start = self._safe_get(c, "start", None)
                end   = self._safe_get(c, "end",   None)

                if ch is None or start is None or end is None:
                    continue

                # timestamp['char'] と token_confidence は同じ長さのはずなのでインデックス対応
                conf = token_conf_list[i] if i < len(token_conf_list) else None

                adjusted_chars.append(
                    {
                        "char": ch,
                        "start": float(start) + segment_start_sec,
                        "end":   float(end)   + segment_start_sec,
                        "confidence": conf,
                    }
                )

            # word も同様にインデックスで結合
            adjusted_words = []
            for i, w in enumerate(words):
                word_text = self._safe_get(w, "word",  None)
                start     = self._safe_get(w, "start", None)
                end       = self._safe_get(w, "end",   None)

                if word_text is None or start is None or end is None:
                    continue

                conf = word_conf_list[i] if i < len(word_conf_list) else None

                adjusted_words.append(
                    {
                        "word":  word_text,
                        "start": float(start) + segment_start_sec,
                        "end":   float(end)   + segment_start_sec,
                        "confidence": conf,
                    }
                )

            adjusted_segments = self._normalize_timestamp_items(
                segments, segment_start_sec, text_key="segment"
            )

            # utterance confidence = token_confidence の平均（score は log-prob なので使わない）
            utterance_confidence = None
            if token_conf_list:
                vals = [v for v in token_conf_list if v is not None]
                if vals:
                    utterance_confidence = sum(vals) / len(vals)
            # fallback: word confidence の平均
            if utterance_confidence is None and word_conf_list:
                vals = [v for v in word_conf_list if v is not None]
                if vals:
                    utterance_confidence = sum(vals) / len(vals)

            return {
                "text": raw_text,
                "start": segment_start_sec,
                "end": segment_end_sec,
                "confidence": float(utterance_confidence) if utterance_confidence is not None else None,
                "char_timestamps": adjusted_chars,
                "word_timestamps": adjusted_words,
                "segment_timestamps": adjusted_segments,
                "debug_hypothesis_keys": sorted(
                    [
                        k for k in dir(first_result)
                        if not k.startswith("_")
                    ]
                )[:200],
            }

        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)