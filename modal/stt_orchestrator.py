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

COMPARE_MERGE_MODEL = "gemini-2.5-flash"
RESCUE_MIN_LEN_SEC = 1.5
RESCUE_PARALLEL_LIMIT = 8

CUT_KEYWORDS = ["なので", "だから", "ですが", "だけど", "そして", "それで", "さらに", "しかし", "でも", "けど", "ただ", "ちなみに", "ところで", "というのも", "ですね", "でしょう", "ますね", "ですよ", "ましょう"]
PRIMARY_MIN_SILENCE_MS = 500
PRIMARY_MIN_CHUNK_MS = 40 * 1000
PRIMARY_FORCE_MAX_CHUNK_MS = 80 * 1000

TURN_STATS_WINDOW_SEC = 10.0
SHORT_TURN_SEC = 3.0

COVERAGE_MAX_GAP_SEC = 0.25
COVERAGE_MIN_RATIO = 0.45
COVERAGE_MIN_SPEECH_SEC = 1.0

FRAGMENT_MAX_CHARS = 4
FRAGMENT_MAX_DURATION_SEC = 1.2

RESCUE_REGION_MERGE_GAP_SEC = 1.0
RESCUE_REGION_MARGIN_SEC = 2.5
RESCUE_REGION_MAX_LEN_SEC = 15.0
RESCUE_SPEAKER_MARGIN_SEC = 0.5

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

COMPARE_AND_MERGE_PROMPT_TEMPLATE = """あなたは会話文字起こしの統合編集者です。
以下の4つの情報を比較し、指定された rescue 区間だけについて、最も自然で忠実な会話テキストを復元してください。

【使う情報】
1. Speaker diarization
   - どの時間帯に誰が話しているかの基準です。
   - speaker の切り替わり順序は、基本的に diarization を優先してください。

2. Primary Parakeet transcript
   - 単語レベル・日本語表現の自然さの主証拠です。
   - 可能な限り Parakeet の語彙・表現を優先してください。

3. Rescue Parakeet transcript
   - 問題区間だけ短く切り出して再実行した結果です。
   - Primary より自然で、欠落補完に役立つ場合は使ってください。

4. Rescue Kotoba transcript
   - 会話の骨格、短いターン、話者交代の検出に役立つ補助証拠です。
   - ただし単語の誤りや崩れがあり得るため、単語そのものは Parakeet を優先してください。

5. 出力形式の厳守:
   - 絶対にプレーンテキストのみの指定されたフォーマットで出力してください。マークダウンのコードフェンスや「こちらが...」のような話し言葉は含めないでください。

【最重要ルール】
- 事実の創作は禁止です。
- 聞こえていない内容を想像で補完しすぎないでください。
- 単語・日本語表現は Parakeet を優先してください。
- 話者交代や短いターンの存在は Kotoba と diarization を参照してください。
- 同じ内容の重複は削除してください。
- rescue 区間だけを出力してください。
- speaker ラベルは diarization に従ってください。
- 出力形式に従ってください。それ以外のフォーマットや話し言葉は絶対に入れないでください。

【出力形式】
[00:12.3s] SPEAKER_01: 整形されたテキスト。
[00:15.8s] SPEAKER_02: 整形されたテキスト。

【対象 rescue 区間】
start={region_start}
end={region_end}

【Diarization speaker segments】
{speaker_segments_text}

【Primary Parakeet transcript in region】
{primary_text}

【Rescue Parakeet transcript in region】
{rescue_parakeet_text}

【Rescue Kotoba transcript in region】
{rescue_kotoba_text}
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
    from pydub.silence import detect_silence
    import io
    import time
    import asyncio

    try:
        ASRService = modal.Cls.from_name("transcription-asr-v1", "ASRService")
        DiarizationService = modal.Cls.from_name(
            "transcription-diarization-v1",
            "DiarizationService",
        )

        asr = ASRService()
        diar = DiarizationService()

        started_at = time.time()

        # ジョブ状態を「処理中」に更新
        prev_job = dict(job_dict.get(job_id, {}))
        job_dict[job_id] = {
            **prev_job,
            "status": "processing",
            "started_at": started_at,
            "current_elapsed_sec": round(started_at - prev_job.get("created_at", started_at), 2),
        }

        print("========== 🚀 PIPELINE START (Parallel Mode) ==========")

        # 0. 音声データのロードと前処理
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1)

        # ---------------------------------------------------------
        # 1. タスクの準備
        # ---------------------------------------------------------

        # A. Diarization
        diar_task = asyncio.to_thread(
            diar.diarize.remote,
            audio_data=audio_bytes,
            num_speakers=num_speakers,
        )

        # B. Primary ASR cut points
        print("🔍 Detecting silence for primary Parakeet chunking...")
        cut_points, silence_ranges = build_primary_asr_cut_points(
            audio,
            min_silence_len_ms=PRIMARY_MIN_SILENCE_MS,
            min_chunk_ms=PRIMARY_MIN_CHUNK_MS,
            force_max_chunk_ms=PRIMARY_FORCE_MAX_CHUNK_MS,
        )

        asr_tasks = build_asr_tasks_from_cut_points(
            audio=audio,
            cut_points=cut_points,
            asr=asr,
        )

        # ---------------------------------------------------------
        # 2. 並列実行
        # ---------------------------------------------------------
        print(f"🔥 Launching Diarization and {len(asr_tasks)} ASR tasks in parallel...")
        all_results = await asyncio.gather(diar_task, *asr_tasks)

        diar_result = all_results[0]
        asr_results = all_results[1:]

        raw_speaker_segments = diar_result.get("segments", [])
        raw_speech_segments = diar_result.get("speech_segments", [])

        speaker_segments = normalize_speaker_segments(raw_speaker_segments)
        speech_segments = normalize_speech_segments(raw_speech_segments)

        turn_stats = build_turn_statistics(
            speaker_segments=speaker_segments,
            window_sec=TURN_STATS_WINDOW_SEC,
            short_turn_sec=SHORT_TURN_SEC,
        )

        update_job_debug(
            job_id,
            "diarization",
            {
                "summary": {
                    "speaker_segment_count": len(speaker_segments),
                    "speech_segment_count": len(speech_segments),
                    "num_speakers": num_speakers,
                },
                "speaker_segments": speaker_segments,
                "speech_segments": speech_segments,
                "turn_stats": turn_stats,
            },
        )

        # ---------------------------------------------------------
        # 3. Primary ASR post-process
        # ---------------------------------------------------------
        all_char_timestamps = []
        for res in asr_results:
            all_char_timestamps.extend(res.get("char_timestamps", []))
        all_char_timestamps.sort(key=lambda x: x["start"])

        coverage_intervals = build_char_coverage_intervals(
            all_char_timestamps,
            max_gap_sec=COVERAGE_MAX_GAP_SEC,
        )

        asr_debug_chunks = build_asr_debug_chunks(asr_results)
        update_job_debug(
            job_id,
            "asr",
            {
                "summary": {
                    "chunk_count": len(asr_results),
                    "char_timestamp_count": len(all_char_timestamps),
                    "cut_point_count": len(cut_points),
                    "silence_count": len(silence_ranges),
                },
                "chunks": asr_debug_chunks,
                "coverage_intervals": coverage_intervals,
            },
        )

        # ---------------------------------------------------------
        # 4. Primary fusion
        # ---------------------------------------------------------
        transcript = merge_chars_and_speakers_with_late_fusion(
            all_char_timestamps,
            speaker_segments,
        )

        transcript = merge_small_transcript_segments(transcript)

        fusion_plain_text = build_fusion_plain_text(transcript)

        update_job_debug(
            job_id,
            "fusion",
            {
                "summary": {
                    "transcript_items": len(transcript),
                    "plain_text_chars": len(fusion_plain_text),
                },
                "transcript": transcript,
                "plain_text": fusion_plain_text,
            },
        )

        print(f"✅ Merged into {len(transcript)} transcript segments.")

        # ---------------------------------------------------------
        # 5. Rescue regions の特定 (欠落 ＆ 短いターン)
        # ---------------------------------------------------------
        print("🔍 Detecting Rescue Regions...")
        
        # A. 文字起こしが抜けている区間
        coverage_gap_regions = detect_coverage_gap_regions(
            speaker_segments=speaker_segments,
            speech_segments=speech_segments,
            coverage_intervals=coverage_intervals,
            min_ratio=COVERAGE_MIN_RATIO,
            min_speech_sec=COVERAGE_MIN_SPEECH_SEC,
        )
        
        # B. 短い会話がポンポン入れ替わる区間
        short_turn_regions = detect_short_turn_regions(
            speaker_segments=speaker_segments,
            window_sec=TURN_STATS_WINDOW_SEC,
        )
        
        # 2つを結合してマージする
        combined_candidate_regions = coverage_gap_regions + short_turn_regions
        raw_rescue_regions = merge_time_regions(combined_candidate_regions, merge_gap_sec=RESCUE_REGION_MERGE_GAP_SEC)
        
        # 前後に少し余裕(マージン)を持たせる
        rescue_regions = [
            expand_region_with_limits(r, len(audio) / 1000.0, margin_sec=RESCUE_REGION_MARGIN_SEC)
            for r in raw_rescue_regions
        ]

        update_job_debug(
            job_id,
            "rescue_detection",
            {
                "summary": {
                    "speaker_segment_region_count": len(rescue_regions),
                },
                "short_turn_regions": [],
                "coverage_gap_regions": [],
                "fragment_regions": [],
                "merged_regions": rescue_regions,
            },
        )

        # ---------------------------------------------------------
        # 6. Rescue ASR (Parakeet re-run + Kotoba)
        # ---------------------------------------------------------
        KotobaWhisperService = modal.Cls.from_name(
            "transcription-kotoba-v1",
            "KotobaWhisperService",
        )
        kotoba = KotobaWhisperService()

        rescue_regions = [
            r for r in rescue_regions
            if r["duration"] >= RESCUE_MIN_LEN_SEC
        ]

        rescue_region_results = await run_rescue_asr_for_regions(
            audio=audio,
            rescue_regions=rescue_regions,
            speaker_segments=speaker_segments,
            asr=asr,
            kotoba=kotoba,
            parallel_limit=RESCUE_PARALLEL_LIMIT,
        )

        update_job_debug(
            job_id,
            "rescue_asr",
            {
                "summary": {
                    "region_count": len(rescue_region_results),
                },
                "regions": rescue_region_results,
            },
        )

        # ---------------------------------------------------------
        # 7. Compare & Merge for rescue regions (parallel)
        # ---------------------------------------------------------
        compare_merge_results = await run_compare_and_merge_for_regions(
            job_id=job_id,
            rescue_region_results=rescue_region_results,
            base_transcript=transcript,
            speaker_segments=speaker_segments,
            parallel_limit=4,
        )

        rewritten_regions = [
            {
                "start": item["region"]["start"],
                "end": item["region"]["end"],
                "transcript": item["merged_transcript"],
            }
            for item in compare_merge_results
        ]

        # ---------------------------------------------------------
        # 8. Replace rescue regions into base transcript
        # ---------------------------------------------------------
        if rewritten_regions:
            transcript = replace_regions_in_transcript(
                base_transcript=transcript,
                rewritten_regions=rewritten_regions,
            )

            fusion_plain_text = build_fusion_plain_text(transcript)

            update_job_debug(
                job_id,
                "fusion",
                {
                    "summary": {
                        "transcript_items": len(transcript),
                        "plain_text_chars": len(fusion_plain_text),
                        "rescue_rewritten_region_count": len(rewritten_regions),
                    },
                    "transcript": transcript,
                    "plain_text": fusion_plain_text,
                },
            )

        # ---------------------------------------------------------
        # 4. LLM整形
        # ---------------------------------------------------------
        print("🪄 Running LLM post-processing...")
        cleaned_transcript = await asyncio.to_thread(
            llm_clean_transcript,
            job_id,
            transcript,
        )
        cleaned_text = transcript_json_to_plain_text(cleaned_transcript)

        # ---------------------------------------------------------
        # 5. 結果の保存
        # ---------------------------------------------------------
        finished_at = time.time()
        created_at = prev_job.get("created_at", started_at)

        final_job = dict(job_dict.get(job_id, {}))
        debug_payload = final_job.get("debug", {})

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
                "debug": debug_payload,
            },
            "debug": debug_payload,
        }
        print("========== ✨ PIPELINE COMPLETE ==========")

    except Exception as e:
        print("========== PIPELINE DEBUG: ERROR ==========")
        print(repr(e))
        print("===========================================")

        import traceback
        traceback.print_exc()

        finished_at = time.time()
        prev_job = dict(job_dict.get(job_id, {}))
        created_at = prev_job.get("created_at", finished_at)
        debug_payload = prev_job.get("debug", {})

        job_dict[job_id] = {
            "status": "error",
            "message": str(e),
            "created_at": created_at,
            "started_at": prev_job.get("started_at"),
            "finished_at": finished_at,
            "elapsed_sec": round(finished_at - created_at, 2),
            "current_elapsed_sec": round(finished_at - created_at, 2),
            "debug": debug_payload,
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
        "debug": {
            "asr": {
                "summary": {},
                "chunks": [],
                "coverage_intervals": [],
            },
            "diarization": {
                "summary": {},
                "speaker_segments": [],
                "speech_segments": [],
                "turn_stats": {},
            },
            "fusion": {
                "summary": {},
                "transcript": [],
                "plain_text": "",
            },
            "rescue_detection": {
                "summary": {},
                "short_turn_regions": [],
                "coverage_gap_regions": [],
                "fragment_regions": [],
                "merged_regions": [],
            },
            "rescue_asr": {
                "summary": {},
                "regions": [],
            },
            "compare_merge": {
                "summary": {},
                "regions": [],
            },
            "llm": {
                "summary": {},
                "inputs": [],
                "outputs": [],
            },
        },
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


def call_llm_for_chunk(job_id: str, index: int, total: int, chunk: dict) -> dict:
    from google import genai
    from google.genai import types

    context_text = "\n".join(chunk["context_lines"]) if chunk["context_lines"] else "(なし)"
    target_text = "\n".join(chunk["target_lines"])

    prompt = PROMPT_TEMPLATE.format(
        context_lines=context_text,
        target_lines=target_text,
    )

    append_job_debug_list(
        job_id,
        "llm",
        "inputs",
        {
            "index": index,
            "context_lines": chunk["context_lines"],
            "target_lines": chunk["target_lines"],
            "prompt_chars": len(prompt),
        },
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

        append_job_debug_list(
            job_id,
            "llm",
            "outputs",
            {
                "index": index,
                "raw_text": cleaned_text,
                "response_chars": len(cleaned_text),
                "elapsed_sec": elapsed,
            },
        )

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


def llm_clean_transcript(job_id: str, transcript: list[dict]) -> list[dict]:
    original_lines = transcript_to_lines(transcript)
    chunks = chunk_lines(
        original_lines,
        target_size=TARGET_LINES,
        context_size=CONTEXT_LINES,
    )

    log_transcript_stats(transcript, original_lines, chunks)

    async def _runner():
        tasks = [
            asyncio.to_thread(call_llm_for_chunk, job_id, i, len(chunks), chunk)
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

    update_job_debug(
        job_id,
        "llm",
        {
            "summary": {
                "input_chunk_count": len(chunks),
                "output_chunk_count": len(chunk_results),
                "cleaned_full_text_chars": len(cleaned_full_text),
                "parsed_items": len(parsed),
            }
        },
    )

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

def merge_chars_and_speakers_with_late_fusion(char_timestamps, speaker_segments):
    chunks = []
    current_chars = []
    current_text = ""
    
    for i, char_info in enumerate(char_timestamps):
        char_text = "".join(char_info["char"])
        
        # 🌟 前の文字とのGap（ポーズ）計算
        gap = 0
        if current_chars:
            gap = char_info["start"] - current_chars[-1]["end"]
        
        # 🌟 ポーズの挿入ロジック
        if gap >= 0.3:
            current_text += " ... "
        current_text += char_text
        current_chars.append(char_info)
        
        # 🌟 分割判定（3つの鉄則）
        is_split = False
        
        # ルール1: 5文字以上かつ「。」がある
        if len(current_text) >= 5 and char_text in ["。", "？", "！"]:
            is_split = True
            
        # ルール2: 無音1秒以上
        elif gap >= 1.0:
            is_split = True
            
        # ルール3: 100文字超えかつ「、」か接続詞
        elif len(current_text) >= 100:
            if char_text == "、" or any(current_text.endswith(k) for k in CUT_KEYWORDS):
                is_split = True

        # 分割実行
        if is_split or i == len(char_timestamps) - 1:
            chunk_start = current_chars[0]["start"]
            chunk_end = current_chars[-1]["end"]
            
            # 🌟 多数決（Majority Vote）による話者判定
            speaker_votes = {}
            for seg in speaker_segments:
                # チャンクと話者セグメントの重なり（秒数）を計算
                overlap_start = max(chunk_start, seg["start"])
                overlap_end = min(chunk_end, seg["end"])
                overlap_dur = max(0, overlap_end - overlap_start)
                
                if overlap_dur > 0:
                    speaker_votes[seg["speaker"]] = speaker_votes.get(seg["speaker"], 0) + overlap_dur
            
            # 最も占有時間の長い話者を採用
            best_speaker = max(speaker_votes, key=speaker_votes.get) if speaker_votes else "UNKNOWN"
            
            chunks.append({
                "speaker": best_speaker,
                "text": current_text.strip(),
                "start": chunk_start,
                "end": chunk_end
            })
            
            # バッファをリセット
            current_chars = []
            current_text = ""
            
    return chunks

def update_job_debug(job_id: str, section: str, payload: dict):
    job = dict(job_dict.get(job_id, {}))
    debug = dict(job.get("debug", {}))
    current_section = dict(debug.get(section, {}))
    current_section.update(payload)
    debug[section] = current_section
    job["debug"] = debug
    job_dict[job_id] = job


def append_job_debug_list(job_id: str, section: str, list_name: str, item: dict):
    job = dict(job_dict.get(job_id, {}))
    debug = dict(job.get("debug", {}))
    current_section = dict(debug.get(section, {}))
    current_list = list(current_section.get(list_name, []))
    current_list.append(item)
    current_section[list_name] = current_list
    debug[section] = current_section
    job["debug"] = debug
    job_dict[job_id] = job


def build_asr_debug_chunks(asr_results: list[dict]) -> list[dict]:
    chunks = []
    for i, res in enumerate(asr_results):
        char_timestamps = res.get("char_timestamps", [])
        word_timestamps = res.get("word_timestamps", [])
        chunks.append({
            "index": i,
            "start": res.get("start"),
            "end": res.get("end"),
            "text": res.get("text") or res.get("raw_text", ""),
            "confidence": res.get("confidence"),
            "char_count": len(char_timestamps),
            "word_count": len(word_timestamps),
            "char_timestamps": char_timestamps,
            "word_timestamps": word_timestamps,
        })
    return chunks

def build_fusion_plain_text(transcript: list[dict]) -> str:
    return "\n".join(
        f"[{format_seconds(float(item['start']))}] {item['speaker']}: {item['text']}"
        for item in transcript
    )


def build_primary_asr_cut_points(
    audio,
    min_silence_len_ms: int = 500,
    min_chunk_ms: int = 10_000,
    force_max_chunk_ms: int = 20_000,
):
    from pydub.silence import detect_silence

    silences = detect_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=audio.dBFS - 16,
    )

    cut_points = [0]
    current_start = 0
    audio_len = len(audio)

    while current_start < audio_len:
        hard_end = min(current_start + force_max_chunk_ms, audio_len)

        candidate_midpoints = []
        for sil_start, sil_end in silences:
            mid = int((sil_start + sil_end) / 2)
            if current_start + min_chunk_ms <= mid <= hard_end:
                candidate_midpoints.append(mid)

        if candidate_midpoints:
            next_cut = candidate_midpoints[-1]
        else:
            next_cut = hard_end

        if next_cut <= current_start:
            break

        cut_points.append(next_cut)
        current_start = next_cut

        if current_start >= audio_len:
            break

    if cut_points[-1] != audio_len:
        cut_points.append(audio_len)

    cut_points = sorted(set(int(x) for x in cut_points))

    return cut_points, silences


def build_asr_tasks_from_cut_points(audio, cut_points, asr):
    tasks = []

    for i in range(len(cut_points) - 1):
        start_ms = cut_points[i]
        end_ms = cut_points[i + 1]
        if end_ms <= start_ms:
            continue

        chunk = audio[start_ms:end_ms]
        buf = io.BytesIO()
        chunk.export(buf, format="wav")

        tasks.append(
            asyncio.to_thread(
                asr.transcribe_segment.remote,
                segment_data=buf.getvalue(),
                segment_start_sec=start_ms / 1000.0,
                segment_end_sec=end_ms / 1000.0,
            )
        )

    return tasks

def normalize_speaker_segments(segments: list[dict]) -> list[dict]:
    normalized = []
    for seg in segments or []:
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue

        normalized.append(
            {
                "start": start,
                "end": end,
                "speaker": seg["speaker"],
            }
        )

    normalized.sort(key=lambda x: (x["start"], x["end"]))
    return normalized


def normalize_speech_segments(segments: list[dict]) -> list[dict]:
    normalized = []
    for seg in segments or []:
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue

        normalized.append(
            {
                "start": start,
                "end": end,
            }
        )

    normalized.sort(key=lambda x: (x["start"], x["end"]))
    return normalized


def build_turn_statistics(
    speaker_segments: list[dict],
    window_sec: float = 10.0,
    short_turn_sec: float = 3.0,
) -> dict:
    if not speaker_segments:
        return {
            "total_segments": 0,
            "short_turn_count": 0,
            "speaker_change_count": 0,
            "max_changes_in_window": 0,
        }

    short_turn_count = sum(
        1
        for seg in speaker_segments
        if (seg["end"] - seg["start"]) <= short_turn_sec
    )

    speaker_change_count = max(0, len(speaker_segments) - 1)

    max_changes_in_window = 0
    for i, seg in enumerate(speaker_segments):
        start = seg["start"]
        end = start + window_sec

        changes = 0
        prev_speaker = None
        for other in speaker_segments[i:]:
            if other["start"] > end:
                break
            if prev_speaker is not None and other["speaker"] != prev_speaker:
                changes += 1
            prev_speaker = other["speaker"]

        max_changes_in_window = max(max_changes_in_window, changes)

    return {
        "total_segments": len(speaker_segments),
        "short_turn_count": short_turn_count,
        "speaker_change_count": speaker_change_count,
        "max_changes_in_window": max_changes_in_window,
    }

def build_char_coverage_intervals(
    all_char_timestamps: list[dict],
    max_gap_sec: float = 0.25,
) -> list[dict]:
    if not all_char_timestamps:
        return []

    chars = sorted(all_char_timestamps, key=lambda x: x["start"])

    intervals = []
    cur_start = float(chars[0]["start"])
    cur_end = float(chars[0]["end"])

    for item in chars[1:]:
        start = float(item["start"])
        end = float(item["end"])

        if start - cur_end <= max_gap_sec:
            cur_end = max(cur_end, end)
        else:
            intervals.append({"start": cur_start, "end": cur_end})
            cur_start = start
            cur_end = end

    intervals.append({"start": cur_start, "end": cur_end})
    return intervals


def calc_interval_overlap(start_a, end_a, start_b, end_b) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def calc_coverage_ratio(region_start, region_end, coverage_intervals: list[dict]) -> float:
    region_len = max(0.0, region_end - region_start)
    if region_len <= 0:
        return 0.0

    covered = 0.0
    for interval in coverage_intervals:
        covered += calc_interval_overlap(
            region_start,
            region_end,
            interval["start"],
            interval["end"],
        )

    return covered / region_len

def detect_coverage_gap_regions(
    speaker_segments: list[dict],
    speech_segments: list[dict],
    coverage_intervals: list[dict],
    min_ratio: float = 0.45,
    min_speech_sec: float = 1.0,
) -> list[dict]:
    candidate_regions = []

    # speech_segments を優先し、なければ speaker_segments も参照
    base_regions = speech_segments if speech_segments else [
        {"start": s["start"], "end": s["end"]}
        for s in speaker_segments
    ]

    for seg in base_regions:
        start = float(seg["start"])
        end = float(seg["end"])
        duration = end - start

        if duration < min_speech_sec:
            continue

        ratio = calc_coverage_ratio(start, end, coverage_intervals)
        if ratio < min_ratio:
            candidate_regions.append(
                {
                    "start": start,
                    "end": end,
                    "reason": "coverage_gap",
                    "coverage_ratio": round(ratio, 4),
                }
            )

    return merge_time_regions(candidate_regions, merge_gap_sec=1.0)


def detect_short_turn_regions(
    speaker_segments: list[dict],
    window_sec: float = 10.0,
    min_changes: int = 3,
    merge_gap_sec: float = 1.0,
) -> list[dict]:
    if not speaker_segments:
        return []

    regions = []

    for i in range(len(speaker_segments)):
        window_start = float(speaker_segments[i]["start"])
        window_end = window_start + window_sec

        included = []
        for seg in speaker_segments[i:]:
            if float(seg["start"]) > window_end:
                break
            included.append(seg)

        if len(included) < 2:
            continue

        changes = 0
        prev_speaker = included[0]["speaker"]
        for seg in included[1:]:
            if seg["speaker"] != prev_speaker:
                changes += 1
            prev_speaker = seg["speaker"]

        if changes >= min_changes:
            regions.append(
                {
                    "start": window_start,
                    "end": min(window_end, float(included[-1]["end"])),
                    "reason": "short_turn",
                    "speaker_change_count": changes,
                }
            )

    return merge_time_regions(regions, merge_gap_sec=merge_gap_sec)

def detect_fragment_regions(
    transcript: list[dict],
    max_chars: int = 4,
    max_duration_sec: float = 1.2,
) -> list[dict]:
    regions = []

    for item in transcript:
        text = (item.get("text") or "").strip()
        duration = float(item["end"]) - float(item["start"])

        is_tiny = len(text) <= max_chars or duration <= max_duration_sec
        looks_fragmented = text in {"オ", "オ。", "はい", "えっと", "じゃあ"} or text.endswith("。") and len(text) <= max_chars

        if is_tiny or looks_fragmented:
            regions.append(
                {
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                    "reason": "fragment",
                    "text": text,
                }
            )

    return merge_time_regions(regions, merge_gap_sec=0.8)


def merge_time_regions(regions: list[dict], merge_gap_sec: float = 1.0) -> list[dict]:
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda x: (x["start"], x["end"]))
    merged = [sorted_regions[0].copy()]

    for region in sorted_regions[1:]:
        last = merged[-1]

        if region["start"] <= last["end"] + merge_gap_sec:
            last["end"] = max(last["end"], region["end"])

            reasons = set(last.get("reasons", []))
            if "reason" in last:
                reasons.add(last["reason"])
                last.pop("reason", None)

            if "reason" in region:
                reasons.add(region["reason"])

            for r in region.get("reasons", []):
                reasons.add(r)

            last["reasons"] = sorted(reasons)
        else:
            copied = region.copy()
            if "reason" in copied:
                copied["reasons"] = [copied.pop("reason")]
            merged.append(copied)

    for region in merged:
        region["start"] = float(region["start"])
        region["end"] = float(region["end"])

    return merged


def expand_region_with_limits(
    region: dict,
    audio_duration_sec: float,
    margin_sec: float = 2.5,
    max_len_sec: float = 15.0,
) -> dict:
    start = max(0.0, float(region["start"]) - margin_sec)
    end = min(audio_duration_sec, float(region["end"]) + margin_sec)

    if end - start > max_len_sec:
        center = (start + end) / 2
        half = max_len_sec / 2
        start = max(0.0, center - half)
        end = min(audio_duration_sec, center + half)

    new_region = dict(region)
    new_region["start"] = round(start, 3)
    new_region["end"] = round(end, 3)
    new_region["duration"] = round(end - start, 3)
    return new_region

def build_rescue_regions_from_speaker_segments(
    speaker_segments: list[dict],
    audio_duration_sec: float,
    margin_sec: float = 0.5,
) -> list[dict]:
    regions = []

    for i, seg in enumerate(speaker_segments):
        raw_start = float(seg["start"])
        raw_end = float(seg["end"])

        if raw_end <= raw_start:
            continue

        start = max(0.0, raw_start - margin_sec)
        end = min(audio_duration_sec, raw_end + margin_sec)

        regions.append(
            {
                "index": i,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "reason": "speaker_segment",
                "speaker": seg["speaker"],
                "source_start": raw_start,
                "source_end": raw_end,
                "source_duration": round(raw_end - raw_start, 3),
            }
        )

    return regions

def slice_audio_region_to_wav_bytes(audio, start_sec: float, end_sec: float) -> bytes:
    start_ms = max(0, int(start_sec * 1000))
    end_ms = min(len(audio), int(end_sec * 1000))
    if end_ms <= start_ms:
        return b""

    chunk = audio[start_ms:end_ms]
    buf = io.BytesIO()
    chunk.export(buf, format="wav")
    return buf.getvalue()

def filter_transcript_by_time_range(
    transcript: list[dict],
    start_sec: float,
    end_sec: float,
) -> list[dict]:
    results = []
    for item in transcript:
        overlap = calc_interval_overlap(
            float(item["start"]),
            float(item["end"]),
            start_sec,
            end_sec,
        )
        if overlap > 0:
            results.append(item.copy())
    return results


def filter_speaker_segments_by_time_range(
    speaker_segments: list[dict],
    start_sec: float,
    end_sec: float,
) -> list[dict]:
    results = []
    for seg in speaker_segments:
        overlap = calc_interval_overlap(
            float(seg["start"]),
            float(seg["end"]),
            start_sec,
            end_sec,
        )
        if overlap > 0:
            results.append(seg.copy())
    return results

def build_transcript_from_kotoba_segments(segments: list[dict]) -> list[dict]:
    transcript = []
    for seg in segments or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        transcript.append(
            {
                "speaker": "UNKNOWN",
                "text": text,
                "start": float(seg["start"]),
                "end": float(seg["end"]),
            }
        )

    transcript.sort(key=lambda x: (x["start"], x["end"]))
    return transcript

async def run_rescue_asr_for_regions(
    audio,
    rescue_regions: list[dict],
    speaker_segments: list[dict], # 🌟 引数を追加
    asr,
    kotoba,
    parallel_limit: int = 8,
) -> list[dict]:
    semaphore = asyncio.Semaphore(parallel_limit)

    async def _run_one(region: dict):
        start_sec = float(region["start"])
        end_sec = float(region["end"])
        
        # ① Kotobaには、レスキュー区間「全体」を一塊で投げる (文脈と全体の骨格を取るため)
        async def _run_kotoba():
            async with semaphore:
                wav_bytes = slice_audio_region_to_wav_bytes(audio, start_sec, end_sec)
                if not wav_bytes:
                    return None

                res = await asyncio.to_thread(
                    kotoba.transcribe_segment.remote,
                    segment_data=wav_bytes,
                    segment_start_sec=start_sec,
                    segment_end_sec=end_sec,
                    language="ja",
                    task="transcribe",
                )
                return res
        
        kotoba_task = asyncio.create_task(_run_kotoba())

        # ② Parakeetには、speaker_segments に従って「細かくスライス」して投げる
        segs_in_region = filter_speaker_segments_by_time_range(speaker_segments, start_sec, end_sec)
        if not segs_in_region:
            segs_in_region = [{"start": start_sec, "end": end_sec, "speaker": "UNKNOWN"}]

        async def _run_parakeet_seg(seg: dict):
            async with semaphore:
                s_start = max(start_sec, float(seg["start"]))
                s_end = min(end_sec, float(seg["end"]))
                
                # 短すぎるノイズは無視
                if s_end - s_start < 0.2:
                    return []
                    
                wav_bytes = slice_audio_region_to_wav_bytes(audio, s_start, s_end)
                if not wav_bytes:
                    return []

                res = await asyncio.to_thread(
                    asr.transcribe_segment.remote,
                    segment_data=wav_bytes,
                    segment_start_sec=s_start,
                    segment_end_sec=s_end,
                )
                return res.get("char_timestamps", [])

        # 細切れにしたタスクを並列で一気に投げる！
        parakeet_tasks = [asyncio.create_task(_run_parakeet_seg(s)) for s in segs_in_region]

        # 両方の完了を待つ
        rescue_kotoba_result = await kotoba_task
        parakeet_char_results = await asyncio.gather(*parakeet_tasks)

        # 細切れになった Parakeet の char_timestamps を1つに統合する
        merged_parakeet_chars = []
        for chars in parakeet_char_results:
            merged_parakeet_chars.extend(chars)
        merged_parakeet_chars.sort(key=lambda x: x["start"])

        return {
            "region": region,
            "rescue_parakeet": {"char_timestamps": merged_parakeet_chars},
            "rescue_kotoba": rescue_kotoba_result, # Kotobaは結果全体をそのまま渡す
        }

    return await asyncio.gather(*[_run_one(r) for r in rescue_regions])


def transcript_to_plain_lines(transcript: list[dict]) -> str:
    if not transcript:
        return "(empty)"
    return "\n".join(
        f"[{format_seconds(float(item['start']))}] {item['speaker']}: {item['text']}"
        for item in transcript
    )


def speaker_segments_to_plain_lines(segments: list[dict]) -> str:
    if not segments:
        return "(empty)"
    return "\n".join(
        f"[{format_seconds(float(seg['start']))} - {format_seconds(float(seg['end']))}] {seg['speaker']}"
        for seg in segments
    )

def llm_compare_and_merge_region(
    job_id: str,
    region_result: dict,
    base_transcript: list[dict],
    speaker_segments: list[dict],
) -> list[dict]:
    from google import genai
    from google.genai import types

    region = region_result["region"]
    start_sec = float(region["start"])
    end_sec = float(region["end"])

    primary_region_transcript = filter_transcript_by_time_range(
        base_transcript,
        start_sec,
        end_sec,
    )

    rescue_parakeet_transcript = []
    rescue_parakeet_result = region_result.get("rescue_parakeet")
    if rescue_parakeet_result:
        rescue_chars = rescue_parakeet_result.get("char_timestamps", [])
        rescue_parakeet_transcript = merge_chars_and_speakers_with_late_fusion(
            rescue_chars,
            filter_speaker_segments_by_time_range(speaker_segments, start_sec, end_sec),
        )
        rescue_parakeet_transcript = merge_small_transcript_segments(rescue_parakeet_transcript)

    rescue_kotoba_transcript = build_transcript_from_kotoba_segments(
        (region_result.get("rescue_kotoba") or {}).get("segments", [])
    )

    region_speaker_segments = filter_speaker_segments_by_time_range(
        speaker_segments,
        start_sec,
        end_sec,
    )

    prompt = COMPARE_AND_MERGE_PROMPT_TEMPLATE.format(
        region_start=format_seconds(start_sec),
        region_end=format_seconds(end_sec),
        speaker_segments_text=speaker_segments_to_plain_lines(region_speaker_segments),
        primary_text=transcript_to_plain_lines(primary_region_transcript),
        rescue_parakeet_text=transcript_to_plain_lines(rescue_parakeet_transcript),
        rescue_kotoba_text=transcript_to_plain_lines(rescue_kotoba_transcript),
    )

    client = genai.Client()
    response = client.models.generate_content(
        model=COMPARE_MERGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            system_instruction=(
                "You are a careful transcript merger. "
                "Prefer Parakeet wording, prefer diarization speaker order, "
                "use Kotoba only as supporting evidence for turn presence and short-turn structure. "
                "Do not invent missing content."
            ),
        ),
    )

    merged_text = response.text or ""

    merged_transcript = parse_cleaned_transcript(
        merged_text,
        primary_region_transcript or rescue_parakeet_transcript or rescue_kotoba_transcript,
    )

    # rescue region の範囲に限定
    merged_transcript = [
        item for item in merged_transcript
        if calc_interval_overlap(
            float(item["start"]),
            float(item["end"]),
            start_sec,
            end_sec,
        ) > 0 or (start_sec <= float(item["start"]) <= end_sec)
    ]

    return merged_transcript

def replace_regions_in_transcript(
    base_transcript: list[dict],
    rewritten_regions: list[dict],
) -> list[dict]:
    if not rewritten_regions:
        return base_transcript

    rewritten_regions = sorted(rewritten_regions, key=lambda x: x["start"])
    result = []

    for item in base_transcript:
        item_start = float(item["start"])
        item_end = float(item["end"])

        overlaps_any = False
        for region in rewritten_regions:
            overlap = calc_interval_overlap(
                item_start,
                item_end,
                float(region["start"]),
                float(region["end"]),
            )
            if overlap > 0:
                overlaps_any = True
                break

        if not overlaps_any:
            result.append(item.copy())

    for region in rewritten_regions:
        for item in region["transcript"]:
            result.append(item.copy())

    result.sort(key=lambda x: (float(x["start"]), float(x["end"])))
    return result

async def run_compare_and_merge_for_regions(
    job_id: str,
    rescue_region_results: list[dict],
    base_transcript: list[dict],
    speaker_segments: list[dict],
    parallel_limit: int = 4,
) -> list[dict]:
    semaphore = asyncio.Semaphore(parallel_limit)

    async def _run_one(region_result: dict):
        async with semaphore:
            merged_region_transcript = await asyncio.to_thread(
                llm_compare_and_merge_region,
                job_id,
                region_result,
                base_transcript,
                speaker_segments,
            )

            return {
                "region": region_result["region"],
                "merged_transcript": merged_region_transcript,
            }

    return await asyncio.gather(*[_run_one(r) for r in rescue_region_results])