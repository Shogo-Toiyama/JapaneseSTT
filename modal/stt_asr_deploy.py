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

        self.asr_model = ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt_ctc-0.6b-ja"
        )
        self.asr_model.to("cuda")
        self.asr_model.eval()

    @modal.method()
    def transcribe_segment(self, segment_data: bytes, segment_start_sec: float, segment_end_sec: float):
        import os
        import tempfile

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
            raw_text = first_result.text if hasattr(first_result, "text") else str(first_result)
            chars = getattr(first_result, "timestamp", {}).get("char", [])

            adjusted_chars = []
            for c in chars:
                adjusted_chars.append(
                    {
                        "char": c["char"],
                        "start": float(c["start"]) + segment_start_sec,
                        "end": float(c["end"]) + segment_start_sec,
                    }
                )

            return {
                "text": raw_text,
                "start": segment_start_sec,
                "end": segment_end_sec,
                "char_timestamps": adjusted_chars,
            }

        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)