import os
import tempfile
import modal

app = modal.App("transcription-reazon-k2-v1")

MODEL_REPO = "reazon-research/reazonspeech-k2-v2"


def download_reazon_k2_model():
    from huggingface_hub import hf_hub_download

    files = [
        "encoder-epoch-99-avg-1.int8.onnx",
        "decoder-epoch-99-avg-1.int8.onnx",
        "joiner-epoch-99-avg-1.int8.onnx",
        "tokens.txt",
    ]

    print("Downloading ReazonSpeech k2 v2 int8 ONNX files...")
    for filename in files:
        path = hf_hub_download(repo_id=MODEL_REPO, filename=filename)
        print("downloaded:", path)
    print("ReazonSpeech k2 v2 int8 cached.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "sherpa-onnx",
        "huggingface_hub",
        "soundfile",
        "numpy",
    )
    .run_function(download_reazon_k2_model)
)


@app.cls(
    image=image,
    # CPUでも動く想定。まずはコスト軽めで試す
    timeout=3600,
    scaledown_window=2,
)
class ReazonK2ASRService:
    @modal.enter()
    def load_model(self):
        import os
        from huggingface_hub import hf_hub_download
        import sherpa_onnx

        encoder = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="encoder-epoch-99-avg-1.int8.onnx",
        )
        decoder = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="decoder-epoch-99-avg-1.int8.onnx",
        )
        joiner = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="joiner-epoch-99-avg-1.int8.onnx",
        )
        tokens = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="tokens.txt",
        )

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
        )

    @modal.method()
    def transcribe_segment(
        self,
        segment_data: bytes,
        segment_start_sec: float = 0.0,
        segment_end_sec: float | None = None,
    ):
        import os
        import tempfile
        import soundfile as sf

        temp_audio_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(segment_data)
                temp_audio.flush()
                temp_audio_path = temp_audio.name

            samples, sample_rate = sf.read(temp_audio_path, dtype="float32")

            if samples.ndim > 1:
                # stereo -> mono
                samples = samples.mean(axis=1)

            if sample_rate != 16000:
                raise ValueError(
                    f"Expected 16kHz WAV input, but got sample_rate={sample_rate}. "
                    "Resample in orchestrator before calling this service."
                )

            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            self.recognizer.decode_stream(stream)

            raw_text = stream.result.text.strip()

            effective_end = (
                float(segment_end_sec)
                if segment_end_sec is not None
                else float(segment_start_sec)
            )

            return {
                "text": raw_text,
                "start": float(segment_start_sec),
                "end": effective_end,
                "segments": [
                    {
                        "index": 0,
                        "start": float(segment_start_sec),
                        "end": effective_end,
                        "text": raw_text,
                    }
                ],
                "plain_text": raw_text,
            }

        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)