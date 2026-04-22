import os
import modal

app = modal.App("transcription-kotoba-v1")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers>=4.39,<5",
        "accelerate",
        "huggingface_hub",
        "numpy",
    )
)


@app.cls(
    image=image,
    gpu="T4",
    timeout=3600,
    scaledown_window=2,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class KotobaWhisperService:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import pipeline
        from huggingface_hub import login

        hf_token = os.environ["HF_TOKEN"]
        login(token=hf_token)

        model_id = "kotoba-tech/kotoba-whisper-v2.2"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            revision="refs/pr/7",
            torch_dtype=dtype,
            device=device,
            model_kwargs=model_kwargs,
            trust_remote_code=True,
        )

    @modal.method()
    def transcribe_segment(
        self,
        segment_data: bytes,
        segment_start_sec: float = 0.0,
        segment_end_sec: float | None = None,
        language: str = "ja",
        task: str = "transcribe",
    ):
        """
        Kotoba-Whisper を ASR 専用で実行し、segment-level timestamps を返す。
        長尺処理は orchestrator 側で分割する想定なので、ここでは chunk_length_s は使わない。

        Returns:
            {
              "start": float,
              "end": float,
              "text": str,
              "segments": [
                  {"start": float, "end": float, "text": str, "raw_chunk": {...}},
                  ...
              ],
              "plain_text": str,
              "raw_result": dict,
            }
        """
        import os
        import tempfile

        temp_audio_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(segment_data)
                temp_audio.flush()
                temp_audio_path = temp_audio.name

            result = self.pipe(
                temp_audio_path,
                return_timestamps=True,
                generate_kwargs={"language": language, "task": task},
            )

            raw_chunks = result.get("chunks", []) or []
            segments = []

            for idx, chunk in enumerate(raw_chunks):
                timestamp = chunk.get("timestamp") or (None, None)
                rel_start = timestamp[0]
                rel_end = timestamp[1]
                text = (chunk.get("text") or "").strip()

                if not text:
                    continue

                if rel_start is None:
                    rel_start = 0.0
                if rel_end is None:
                    rel_end = rel_start

                abs_start = float(segment_start_sec) + float(rel_start)
                abs_end = float(segment_start_sec) + float(rel_end)

                segments.append(
                    {
                        "index": idx,
                        "start": abs_start,
                        "end": abs_end,
                        "relative_start": float(rel_start),
                        "relative_end": float(rel_end),
                        "text": text,
                        "raw_chunk": chunk,
                    }
                )

            plain_text = "\n".join(seg["text"] for seg in segments)

            effective_end = float(segment_end_sec) if segment_end_sec is not None else (
                segments[-1]["end"] if segments else float(segment_start_sec)
            )

            return {
                "start": float(segment_start_sec),
                "end": effective_end,
                "text": result.get("text", "") or plain_text,
                "segments": segments,
                "plain_text": plain_text,
                "raw_result": result,
            }

        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
