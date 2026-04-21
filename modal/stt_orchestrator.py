import io
import os
import re
import uuid
import asyncio
import modal
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
    .pip_install("fastapi", "python-multipart", "pydub", "openai")
)

job_dict = modal.Dict.from_name("stt-jobs-dict", create_if_missing=True)

LLM_MODEL = "openai/gpt-oss-120b"
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
    secrets=[modal.Secret.from_name("deepinfra-secret")]
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

        diar_result = await asyncio.to_thread(
            diar.diarize.remote,
            audio_data=audio_bytes,
            num_speakers=num_speakers,
        )

        speaker_segments = diar_result.get("segments", [])
        speech_segments = diar_result.get("speech_segments", [])

        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1)

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

        transcript = merge_chars_and_speakers_with_gaps(all_char_timestamps, speaker_segments)

        # --- ここから LLM 整形 ---
        cleaned_transcript = await asyncio.to_thread(
            llm_clean_transcript,
            transcript,
        )

        cleaned_text = transcript_json_to_plain_text(cleaned_transcript)

        job_dict[job_id] = {
            "status": "completed",
            "result": {
                "transcript": transcript,
                "cleaned_transcript": cleaned_transcript,
                "cleaned_text": cleaned_text,
            },
        }

    except Exception as e:
        job_dict[job_id] = {
            "status": "error",
            "message": str(e),
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
    job_dict[job_id] = {"status": "processing"}

    process_audio_background.spawn(job_id, audio_bytes, num_speakers)

    return {"job_id": job_id}


@web_app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in job_dict:
        return {"status": "not_found"}
    return job_dict[job_id]


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
            merged.append({
                "speaker": speaker,
                "text": text_chunk,
                "start": char_info["start"],
                "end": char_info["end"]
            })

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


def chunk_lines(lines: list[str], target_size: int = 30, context_size: int = 5) -> list[dict]:
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + target_size, len(lines))
        context_start = max(0, start - context_size)

        chunks.append({
            "context_lines": lines[context_start:start],
            "target_lines": lines[start:end],
        })
        start = end

    return chunks

def transcript_json_to_plain_text(transcript: list[dict]) -> str:
    lines = []
    for item in transcript:
        ts = format_seconds(float(item["start"]))
        lines.append(f"[{ts}] {item['speaker']}: {item['text']}")
    return "\n".join(lines)


def llm_clean_transcript(transcript: list[dict]) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DEEPINFRA_TOKEN"],
        base_url="https://api.deepinfra.com/v1/openai"
    )

    original_lines = transcript_to_lines(transcript)
    chunks = chunk_lines(original_lines, target_size=TARGET_LINES, context_size=CONTEXT_LINES)

    cleaned_text_blocks = []
    for chunk in chunks:
        prompt = PROMPT_TEMPLATE.format(
            context_lines="\n".join(chunk["context_lines"]) if chunk["context_lines"] else "(なし)",
            target_lines="\n".join(chunk["target_lines"]),
        )

        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful transcript editor. Do not add markdown. Do not omit content. Output only the target portion in the specified plain-text format.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
        )

        cleaned_text = resp.choices[0].message.content or ""
        cleaned_text_blocks.append(cleaned_text.strip())

    cleaned_full_text = "\n".join(block for block in cleaned_text_blocks if block.strip())
    return parse_cleaned_transcript(cleaned_full_text, transcript)


LINE_RE = re.compile(
    r"^\[(?P<mm>\d{2}):(?P<ss>\d{2}(?:\.\d)?)s\]\s+(?P<speaker>[^:]+):\s*(?P<text>.+?)\s*$"
)


def parse_cleaned_transcript(cleaned_text: str, original_transcript: list[dict]) -> list[dict]:
    """
    LLM のプレーンテキスト出力を transcript JSON に戻す。
    start は LLM 出力の timestamp を使い、
    end は次の発話開始時刻の直前、最後だけ元 transcript の末尾 end を使う。
    """
    parsed = []
    for raw_line in cleaned_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = LINE_RE.match(line)
        if not m:
            continue

        mm = int(m.group("mm"))
        ss = float(m.group("ss"))
        start = mm * 60 + ss

        parsed.append({
            "speaker": m.group("speaker").strip(),
            "text": m.group("text").strip(),
            "start": start,
            "end": start,  # 仮
        })

    if not parsed:
        return original_transcript

    # end を補完
    for i in range(len(parsed) - 1):
        parsed[i]["end"] = parsed[i + 1]["start"]

    parsed[-1]["end"] = float(original_transcript[-1]["end"]) if original_transcript else parsed[-1]["start"]

    return parsed