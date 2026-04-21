import json
from pathlib import Path
import modal

AUDIO_FILE = "sample_audio.m4a"
NUM_SPEAKERS = 2

APP_ASR = "transcription-asr-v1"
CLS_ASR = "ASRService"

APP_DIAR = "transcription-diarization-v1"
CLS_DIAR = "DiarizationService"


def main():
    audio_path = Path(__file__).resolve().parent / AUDIO_FILE
    output_path = Path(__file__).resolve().parent / "result.json"

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with audio_path.open("rb") as f:
        audio_bytes = f.read()

    ASRService = modal.Cls.from_name(APP_ASR, CLS_ASR)
    DiarizationService = modal.Cls.from_name(APP_DIAR, CLS_DIAR)

    asr = ASRService()
    diar = DiarizationService()

    asr_result = asr.transcribe.remote(audio_data=audio_bytes)
    diar_result = diar.diarize.remote(
        audio_data=audio_bytes,
        num_speakers=NUM_SPEAKERS,
    )

    result = {
        "raw_text": asr_result.get("raw_text", ""),
        "char_timestamps": asr_result.get("char_timestamps", []),
        "word_timestamps": asr_result.get("word_timestamps", []),
        "segment_timestamps": asr_result.get("segment_timestamps", []),
        "speaker_segments": diar_result.get("segments", []),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("saved:", output_path)
    print("text:", result["raw_text"])
    print("segment timestamps:", result["segment_timestamps"][:3])
    print("speaker segments:", result["speaker_segments"][:3])


if __name__ == "__main__":
    main()