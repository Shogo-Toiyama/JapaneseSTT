import io
import os
import modal

app = modal.App("transcription-diarization-v1")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .env({
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    })
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "pyannote.audio==4.0.1",
        "pydub",
        "soundfile",
        "huggingface_hub",
    )
)

@app.cls(
    image=image,
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=2,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class DiarizationService:
    @modal.enter(snap=True)
    def load_model(self):
        import os
        import torch
        from pyannote.audio import Pipeline
        from pyannote.audio.core.task import Specifications

        torch.serialization.add_safe_globals([Specifications])

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=os.environ["HF_TOKEN"],
        )
        self.pipeline.to(torch.device("cuda"))

    @modal.method()
    def diarize(self, audio_data: bytes, num_speakers: int | None = None):
        import io
        import torch
        import soundfile as sf
        from pydub import AudioSegment

        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        audio = audio.set_frame_rate(16000).set_channels(1)

        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        waveform_np, sample_rate = sf.read(wav_buffer, dtype="float32")

        if waveform_np.ndim == 1:
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        else:
            waveform = torch.from_numpy(waveform_np).transpose(0, 1)

        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers

        result = self.pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            },
            **kwargs,
        )

        annotation = getattr(result, "speaker_diarization", result)

        speaker_segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            speaker_segments.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": speaker,
                }
            )

        # 4.x の speech activity / segmentation があれば使う。なければ話者区間を speech とみなす。
        speech_segments = []
        sad_like = getattr(result, "speech_activity", None) or getattr(result, "segmentation", None)

        if sad_like is not None and hasattr(sad_like, "itersegments"):
            for turn in sad_like.itersegments():
                speech_segments.append(
                    {
                        "start": float(turn.start),
                        "end": float(turn.end),
                    }
                )
        else:
            for seg in speaker_segments:
                speech_segments.append(
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                    }
                )

        return {
            "segments": speaker_segments,
            "speech_segments": merge_speech_segments(speech_segments),
        }


def merge_speech_segments(segments, max_gap=0.6, min_duration=0.6):
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x["start"])
    merged = [segments[0].copy()]

    for seg in segments[1:]:
        last = merged[-1]
        if seg["start"] - last["end"] <= max_gap:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(seg.copy())

    return [
        seg
        for seg in merged
        if (seg["end"] - seg["start"]) >= min_duration
    ]