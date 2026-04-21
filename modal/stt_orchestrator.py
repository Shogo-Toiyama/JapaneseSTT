import io
import os
import re
import uuid
import asyncio
import modal
import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = modal.App("transcription-orchestrator-v1")
web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi", "python-multipart", "pydub", "google-genai")
)

job_dict = modal.Dict.from_name("stt-jobs-dict", create_if_missing=True)

# Gemini API で使うモデル名
LLM_MODEL = "gemini-2.5-flash"
TARGET_LINES = 30
CONTEXT_LINES = 5

PROMPT_TEMPLATE = """あなたはプロフェッショナルな議事録作成者であり、優秀な編集者です。
音声認識（ASR）から出力された1行ずつの会話テキストを、読みやすく自然な会話劇に整形してください。

【入力データの仕様】
・各行は `[開始時間] 話者: テキスト` のフォーマットです。
・テキスト内の「...」は、発話中の「物理的な短いポーズ（空白時間・息継ぎ）」を表しています。
・音声認識特有の誤字脱字、同音異義語のミス、文の分断が含まれています。

【整形ルール】
1. 結合と分割（最重要）:
   - インプットは細かく分割されています。文脈を読み取り、不自然な位置で区切られている場合は「自然で意味のある一文」に結合してください。
   - 結合する際、出力のタイムスタンプは「結合したブロックの最初の時間」を必ずそのまま使用してください。
   - 逆に、文章が分かれていると判断した場合は適切な部分で句読点を入れてください。
2. 句読点の補完と「...」の処理:
   - 「...」は、自然な文章の区切りであれば「。」にし、文が続く場合は「、」にするか、自然に削除してください。
   - 「...」がない場所でも、文法や息継ぎとして適切だと判断した箇所には積極的に読点（、）や句点（。）を補ってください。
3. 固有名詞のローマ字化と補正:
   - 音声認識のミス（例：「ラッキー」→「学期」）は文脈から推測して修正してください。
   - 以下の固有名詞は、必ず指定の表記に統一してください。人物名はローマ字に直してください。
     * スマイリング / スマイル / Smiling -> SmiRing
     * しょうご / 将吾 -> Shogo
     * つきね / 月音 -> Tsukine
4. トーン＆マナーの維持:
   - 話者の一人称（私、僕、俺など）、語尾のニュアンス、関西弁などの口調は絶対に標準語化・改変しないでください。
   - 要約は行わず、すべての発言内容を残してください。
5. 出力形式の厳守:
   - 絶対にプレーンテキストのみの指定されたフォーマットで出力してください。マークダウンのコードフェンスや「こちらが...」のような話し言葉は含めないでください。
   - 必ず[Target]に指定された部分だけを出力してください。[Context]の部分は文脈理解のみに使用し、出力には含めないでください。

【出力フォーマット】
[00:12.3s] SPEAKER_01: 整形されたテキスト。
[00:15.8s] SPEAKER_02: 整形されたテキスト。

【処理対象】
[Context]（※以下の数行は文脈把握用です。出力には含めないでください）
{context_lines}

[Target]（※ここから下のテキストを整形して出力してください）
{target_lines}
"""


@app.function(
    image=orchestrator_image,
    timeout=3600,
    secrets=[modal.Secret.from_name("gemini-secret")],
)
async def process_audio_background(
    job_id: str,
    audio_bytes: bytes,
    num_speakers: int | None = None,
):
    from pydub import AudioSegment

    try:
        ASRService = modal.Cls.from_name("transcription-asr-v1", "ASRService")
        DiarizationService = modal.Cls.from_name(
            "transcription-diarization-v1",
            "DiarizationService",
        )

        asr = ASRService()
        diar = DiarizationService()

        started_at = time.time()

        prev_job = job_dict.get(job_id, {})
        job_dict[job_id] = {
            **prev_job,
            "status": "processing",
            "started_at": started_at,
            "current_elapsed_sec": round(started_at - prev_job.get("created_at", started_at), 2),
        }

        print("========== PIPELINE DEBUG: START ==========")
        print(f"job_id={job_id}")
        print(f"audio_bytes_size={len(audio_bytes)}")
        print(f"num_speakers={num_speakers}")
        print(f"started_at={started_at}")
        print("===========================================")

        diar_result = await asyncio.to_thread(
            diar.diarize.remote,
            audio_data=audio_bytes,
            num_speakers=num_speakers,
        )

        speaker_segments = diar_result.get("segments", [])
        speech_segments = diar_result.get("speech_segments", [])

        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1)

        print("========== PIPELINE DEBUG: DIARIZATION ==========")
        print(f"speaker_segments={len(speaker_segments)}")
        print(f"speech_segments={len(speech_segments)}")
        if speaker_segments:
            print("first_speaker_segment=", speaker_segments[0])
            print("last_speaker_segment=", speaker_segments[-1])
        if speech_segments:
            print("first_speech_segment=", speech_segments[0])
            print("last_speech_segment=", speech_segments[-1])
        print("=================================================")

        tasks = []
        for seg in speech_segments:
            start_ms = max(0, int(seg["start"] * 1000))
            end_ms = min(len(audio), int(seg["end"] * 1000))
            if end_ms <= start_ms:
                continue

            chunk = audio[start_ms:end_ms]
            buf = io.BytesIO()
            chunk.export(buf, format="wav")

            tasks.append(
                asyncio.to_thread(
                    asr.transcribe_segment.remote,
                    segment_data=buf.getvalue(),
                    segment_start_sec=seg["start"],
                    segment_end_sec=seg["end"],
                )
            )

        asr_results = await asyncio.gather(*tasks) if tasks else []

        all_char_timestamps = []
        for res in asr_results:
            all_char_timestamps.extend(res.get("char_timestamps", []))
        all_char_timestamps.sort(key=lambda x: x["start"])

        transcript = merge_chars_and_speakers_with_gaps(
            all_char_timestamps,
            speaker_segments,
        )
        transcript = merge_small_transcript_segments(transcript)

        print("========== PIPELINE DEBUG: ASR / MERGE ==========")
        print(f"asr_results={len(asr_results)}")
        print(f"all_char_timestamps={len(all_char_timestamps)}")
        print(f"merged_transcript_items={len(transcript)}")
        if transcript:
            print("first_transcript_item=", transcript[0])
            print("last_transcript_item=", transcript[-1])
        print("=================================================")

        cleaned_transcript = await asyncio.to_thread(
            llm_clean_transcript,
            transcript,
        )

        cleaned_text = transcript_json_to_plain_text(cleaned_transcript)

        print("========== PIPELINE DEBUG: FINAL ==========")
        print(f"cleaned_transcript_items={len(cleaned_transcript)}")
        print(f"cleaned_text_chars={len(cleaned_text)}")
        print("===========================================")

        finished_at = time.time()
        prev_job = job_dict.get(job_id, {})
        created_at = prev_job.get("created_at", started_at)

        job_dict[job_id] = {
            "status": "completed",
            "created_at": created_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_sec": round(finished_at - created_at, 2),
            "current_elapsed_sec": round(finished_at - created_at, 2),
            "result": {
                "transcript": transcript,
                "cleaned_transcript": cleaned_transcript,
                "cleaned_text": cleaned_text,
                "timing": {
                    "created_at": created_at,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_sec": round(finished_at - created_at, 2),
                },
            },
        }

    except Exception as e:
        print("========== PIPELINE DEBUG: ERROR ==========")
        print(repr(e))
        print("===========================================")
        finished_at = time.time()
        prev_job = job_dict.get(job_id, {})
        created_at = prev_job.get("created_at", finished_at)

        job_dict[job_id] = {
            "status": "error",
            "message": str(e),
            "created_at": created_at,
            "started_at": prev_job.get("started_at"),
            "finished_at": finished_at,
            "elapsed_sec": round(finished_at - created_at, 2),
            "current_elapsed_sec": round(finished_at - created_at, 2),
        }


@app.function(
    image=orchestrator_image,
    scaledown_window=2,
)
@modal.asgi_app()
def fastapi_app():
    return web_app


@web_app.post("/transcribe")
async def transcribe_endpoint(
    audio_file: UploadFile = File(...),
    num_speakers: int = Form(None),
):
    audio_bytes = await audio_file.read()

    job_id = str(uuid.uuid4())
    created_at = time.time()

    job_dict[job_id] = {
        "status": "processing",
        "created_at": created_at,
        "started_at": None,
        "finished_at": None,
        "elapsed_sec": None,
        "current_elapsed_sec": 0.0,
    }

    print("========== API DEBUG: NEW JOB ==========")
    print(f"job_id={job_id}")
    print(f"audio_bytes_size={len(audio_bytes)}")
    print(f"num_speakers={num_speakers}")
    print(f"created_at={created_at}")
    print("========================================")

    process_audio_background.spawn(job_id, audio_bytes, num_speakers)

    return {
        "job_id": job_id,
        "created_at": created_at,
    }


@web_app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in job_dict:
        return {"status": "not_found"}

    job = dict(job_dict[job_id])

    created_at = job.get("created_at")
    if created_at and job.get("status") == "processing":
        job["current_elapsed_sec"] = round(time.time() - created_at, 2)

    return job


def merge_chars_and_speakers_with_gaps(char_timestamps, speaker_segments):
    merged = []
    for char_info in char_timestamps:
        text_chunk = "".join(char_info["char"])
        mid_point = (char_info["start"] + char_info["end"]) / 2

        speaker = "UNKNOWN"
        for seg in speaker_segments:
            if seg["start"] <= mid_point <= seg["end"]:
                speaker = seg["speaker"]
                break

        if speaker == "UNKNOWN" and merged:
            speaker = merged[-1]["speaker"]

        gap = 0
        if merged and merged[-1]["speaker"] == speaker:
            gap = char_info["start"] - merged[-1]["end"]

        if merged and merged[-1]["speaker"] == speaker:
            if gap >= 0.3:
                merged[-1]["text"] += " ... " + text_chunk
            else:
                merged[-1]["text"] += text_chunk
            merged[-1]["end"] = char_info["end"]
        else:
            merged.append(
                {
                    "speaker": speaker,
                    "text": text_chunk,
                    "start": char_info["start"],
                    "end": char_info["end"],
                }
            )

    return merged


def format_seconds(sec: float) -> str:
    minutes = int(sec // 60)
    seconds = sec - minutes * 60
    return f"{minutes:02d}:{seconds:04.1f}s"


def transcript_to_lines(transcript: list[dict]) -> list[str]:
    lines = []
    for item in transcript:
        ts = format_seconds(float(item["start"]))
        lines.append(f"[{ts}] {item['speaker']}: {item['text']}")
    return lines


def chunk_lines(
    lines: list[str],
    target_size: int = 30,
    context_size: int = 5,
) -> list[dict]:
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + target_size, len(lines))
        context_start = max(0, start - context_size)

        chunks.append(
            {
                "context_lines": lines[context_start:start],
                "target_lines": lines[start:end],
            }
        )
        start = end

    return chunks


def transcript_json_to_plain_text(transcript: list[dict]) -> str:
    lines = []
    for item in transcript:
        ts = format_seconds(float(item["start"]))
        lines.append(f"[{ts}] {item['speaker']}: {item['text']}")
    return "\n".join(lines)


def debug_preview(text: str, limit: int = 500) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def log_transcript_stats(
    transcript: list[dict],
    lines: list[str],
    chunks: list[dict],
) -> None:
    total_chars = sum(len(item.get("text", "")) for item in transcript)
    total_duration = 0.0
    if transcript:
        total_duration = float(transcript[-1]["end"]) - float(transcript[0]["start"])

    print("========== LLM DEBUG: TRANSCRIPT SUMMARY ==========")
    print(f"transcript_items={len(transcript)}")
    print(f"line_count={len(lines)}")
    print(f"chunk_count={len(chunks)}")
    print(f"total_text_chars={total_chars}")
    print(f"approx_total_duration_sec={total_duration:.2f}")

    if transcript:
        print("first_transcript_item=", transcript[0])
        print("last_transcript_item=", transcript[-1])

    if lines:
        print("first_3_lines:")
        for line in lines[:3]:
            print("  ", line)

    print("===================================================")


def log_chunk_input(idx: int, total: int, chunk: dict, prompt: str) -> None:
    print(f"========== LLM DEBUG: CHUNK INPUT {idx}/{total} ==========")
    print(f"context_line_count={len(chunk['context_lines'])}")
    print(f"target_line_count={len(chunk['target_lines'])}")
    print(f"prompt_chars={len(prompt)}")

    print("context_preview:")
    print(debug_preview("\n".join(chunk["context_lines"]), limit=700))

    print("target_preview:")
    print(debug_preview("\n".join(chunk["target_lines"]), limit=1000))

    print("prompt_preview:")
    print(debug_preview(prompt, limit=1200))
    print("==========================================================")


def log_chunk_output(idx: int, total: int, cleaned_text: str) -> None:
    line_count = len([line for line in cleaned_text.splitlines() if line.strip()])

    print(f"========== LLM DEBUG: CHUNK OUTPUT {idx}/{total} ==========")
    print(f"response_chars={len(cleaned_text)}")
    print(f"response_line_count={line_count}")
    print("response_preview:")
    print(debug_preview(cleaned_text, limit=1200))
    print("===========================================================")


def log_parse_result(
    cleaned_full_text: str,
    parsed: list[dict],
    original_transcript: list[dict],
) -> None:
    print("========== LLM DEBUG: PARSE RESULT ==========")
    print(f"cleaned_full_text_chars={len(cleaned_full_text)}")
    print(f"cleaned_full_text_lines={len([x for x in cleaned_full_text.splitlines() if x.strip()])}")
    print(f"parsed_items={len(parsed)}")
    print(f"original_transcript_items={len(original_transcript)}")

    if parsed:
        print("first_parsed_item=", parsed[0])
        print("last_parsed_item=", parsed[-1])

    print("=============================================")


def call_llm_for_chunk(index: int, total: int, chunk: dict) -> dict:
    from google import genai
    from google.genai import types

    prompt = PROMPT_TEMPLATE.format(
        context_lines="\n".join(chunk["context_lines"]) if chunk["context_lines"] else "(なし)",
        target_lines="\n".join(chunk["target_lines"]),
    )

    log_chunk_input(index, total, chunk, prompt)

    started = time.time()
    print(f"[LLM START] chunk={index}/{total}")

    try:
        client = genai.Client()

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                system_instruction=(
                    "You are a careful transcript editor. "
                    "Do not add markdown. Do not omit content. "
                    "Output only the target portion in the specified plain-text format."
                ),
            ),
        )

        cleaned_text = response.text or ""

        elapsed = round(time.time() - started, 2)
        print(
            f"[LLM DONE] chunk={index}/{total} "
            f"response_chars={len(cleaned_text)} elapsed_sec={elapsed}"
        )

        log_chunk_output(index, total, cleaned_text)

        return {
            "index": index,
            "cleaned_text": cleaned_text,
        }

    except Exception as e:
        elapsed = round(time.time() - started, 2)
        print(
            f"[LLM ERROR] chunk={index}/{total} "
            f"elapsed_sec={elapsed} error={repr(e)}"
        )
        raise


def llm_clean_transcript(transcript: list[dict]) -> list[dict]:
    original_lines = transcript_to_lines(transcript)
    chunks = chunk_lines(
        original_lines,
        target_size=TARGET_LINES,
        context_size=CONTEXT_LINES,
    )

    log_transcript_stats(transcript, original_lines, chunks)

    async def _runner():
        tasks = [
            asyncio.to_thread(call_llm_for_chunk, i, len(chunks), chunk)
            for i, chunk in enumerate(chunks, start=1)
        ]
        return await asyncio.gather(*tasks)

    chunk_results = asyncio.run(_runner())
    chunk_results = sorted(chunk_results, key=lambda x: x["index"])

    cleaned_text_blocks = [
        item["cleaned_text"].strip()
        for item in chunk_results
        if item["cleaned_text"].strip()
    ]

    cleaned_full_text = "\n".join(cleaned_text_blocks)

    parsed = parse_cleaned_transcript(cleaned_full_text, transcript)
    log_parse_result(cleaned_full_text, parsed, transcript)

    return parsed


LINE_RE = re.compile(
    r"^\[(?P<mm>\d{2}):(?P<ss>\d{2}(?:\.\d)?)s\]\s+(?P<speaker>[^:]+):\s*(?P<text>.+?)\s*$"
)


def parse_cleaned_transcript(
    cleaned_text: str,
    original_transcript: list[dict],
) -> list[dict]:
    parsed = []
    unparsable_lines = []

    for raw_line in cleaned_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = LINE_RE.match(line)
        if not m:
            unparsable_lines.append(line)
            continue

        mm = int(m.group("mm"))
        ss = float(m.group("ss"))
        start = mm * 60 + ss

        parsed.append(
            {
                "speaker": m.group("speaker").strip(),
                "text": m.group("text").strip(),
                "start": start,
                "end": start,
            }
        )

    print("========== LLM DEBUG: PARSER ==========")
    print(f"parsed_lines={len(parsed)}")
    print(f"unparsable_lines={len(unparsable_lines)}")
    if unparsable_lines:
        print("unparsable_preview:")
        for line in unparsable_lines[:10]:
            print("  ", line)
    print("=======================================")

    if not parsed:
        print("PARSER FALLBACK: returning original_transcript")
        return original_transcript

    for i in range(len(parsed) - 1):
        parsed[i]["end"] = parsed[i + 1]["start"]

    parsed[-1]["end"] = (
        float(original_transcript[-1]["end"])
        if original_transcript
        else parsed[-1]["start"]
    )
    return parsed

def merge_small_transcript_segments(
    transcript: list[dict],
    min_chars: int = 6,
    min_duration: float = 1.2,
    max_gap_same_speaker: float = 1.0,
) -> list[dict]:
    if not transcript:
        return []

    merged = [transcript[0].copy()]

    for seg in transcript[1:]:
        last = merged[-1]
        gap = seg["start"] - last["end"]
        seg_duration = seg["end"] - seg["start"]

        is_small = (
            len(seg.get("text", "").strip()) <= min_chars
            or seg_duration <= min_duration
        )

        if (
            seg["speaker"] == last["speaker"]
            and (gap <= max_gap_same_speaker or is_small)
        ):
            if last["text"] and seg["text"]:
                last["text"] += " " + seg["text"]
            else:
                last["text"] += seg["text"]
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    # 2パス目: 単独の極小断片を前後に吸収
    final = []
    for i, seg in enumerate(merged):
        text_len = len(seg.get("text", "").strip())
        duration = seg["end"] - seg["start"]
        is_tiny = text_len <= 2 or duration <= 0.6

        if not is_tiny:
            final.append(seg)
            continue

        prev_seg = final[-1] if final else None
        next_seg = merged[i + 1] if i + 1 < len(merged) else None

        if prev_seg and prev_seg["speaker"] == seg["speaker"]:
            prev_seg["text"] += seg["text"]
            prev_seg["end"] = seg["end"]
        elif next_seg and next_seg["speaker"] == seg["speaker"]:
            next_seg["text"] = seg["text"] + next_seg["text"]
            next_seg["start"] = seg["start"]
        elif prev_seg:
            prev_seg["text"] += seg["text"]
            prev_seg["end"] = seg["end"]
        else:
            final.append(seg)

    return final