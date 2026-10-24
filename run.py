#!/usr/bin/env python3
"""ZentroMart Master Execution Script.

Runs the full end-to-end AI Customer Support Agent pipeline on a folder of customer audio files:
1. Gnani STT (en-IN) -> Speech-to-Text transcription.
2. Relational Database & RAG Search -> Retrieves top document chunks with precedence ranking.
3. Multi-Turn Agent Reasoning & Mock Tools (lookup_order, create_ticket) -> Identity checks & policy enforcement.
4. Natural Speech Normalization -> Formats dates, years, and order IDs for speech.
5. Gnani TTS (timbre-v2.5, Kaveri, en-IN) -> Spoken .wav audio reply.

Usage:
    python run.py <audio_folder>
    python run.py --input-dir <audio_folder> --output-dir <output_folder>
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        with open(_env_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.strip().strip("'\"")
                    if _k not in os.environ:
                        os.environ[_k] = _v

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from conversation_agent import ConversationSession, ZentroMartAgent
    from voice_agent import (
        FALLBACK_UNCLEAR_AUDIO_TEXT,
        ZentroMartVoicePipeline,
        format_text_for_tts,
        get_gnani_api_key,
        is_transcript_empty_or_muffled,
    )
except ImportError:
    from src.conversation_agent import ConversationSession, ZentroMartAgent
    from src.voice_agent import (
        FALLBACK_UNCLEAR_AUDIO_TEXT,
        ZentroMartVoicePipeline,
        format_text_for_tts,
        get_gnani_api_key,
    )


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".opus"}


def process_audio_folder(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path] = "output_replies",
    language_code: str = "en-IN",
    voice: str = "Kaveri",
    model: str = "timbre-v2.5",
) -> List[Dict[str, Any]]:
    """Processes every customer audio file in input_folder through the complete agent pipeline.

    Outputs:
    - Text reply
    - Spoken audio reply (.wav)
    - Retrieved document chunks used to answer
    """
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"Input audio folder '{in_dir}' does not exist or is not a directory.")

    # Find all audio files
    audio_files = sorted([f for f in in_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS])

    if not audio_files:
        print(f"⚠️ No audio files found in '{in_dir}'. (Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))})")
        return []

    print("=" * 75)
    print("      ZentroMart AI Customer Support Agent - Audio Batch Processing")
    print("=" * 75)
    print(f"📁 Input Directory  : {in_dir.resolve()}")
    print(f"📁 Output Directory : {out_dir.resolve()}")
    print(f"🎙️  Audio Files Found: {len(audio_files)}")
    print(f"🗣️  STT Language     : {language_code}")
    print(f"🔊 TTS Voice & Model: {voice} ({model}, {language_code})\n")

    # Initialize agent and voice pipeline
    agent = ZentroMartAgent()
    pipeline = ZentroMartVoicePipeline(agent=agent)

    # Maintain conversation session across multi-turn files or per file
    session = ConversationSession(conversation_id=f"BATCH-{int(time.time())}")

    batch_results: List[Dict[str, Any]] = []

    for idx, audio_path in enumerate(audio_files, 1):
        print("-" * 75)
        print(f"[{idx}/{len(audio_files)}] Processing: {audio_path.name}")
        file_start = time.time()

        # -------------------------------------------------------------
        # 1. Speech-to-Text (Gnani STT)
        # -------------------------------------------------------------
        print("  ▶ 1. Running Gnani STT (en-IN)...")
        try:
            transcript = pipeline.transcribe_audio(audio_path, language_code=language_code)
        except Exception as e:
            print(f"  ❌ STT Error: {e}")
            transcript = ""

        print(f"     Customer Spoke: \"{transcript}\"")

        # Check for STT silences or muffled audio: bypass LLM gracefully
        if is_transcript_empty_or_muffled(transcript):
            print("  ⚠️  STT audio silent or muffled. Bypassing LLM with instant fallback...")
            agent_reply = FALLBACK_UNCLEAR_AUDIO_TEXT
            retrieved_chunks = []
            print(f"     Agent Audio Reply: \"{agent_reply}\"")
        else:
            # -------------------------------------------------------------
            # 2. Knowledge Base Retrieval (RAG Chunks)
            # -------------------------------------------------------------
            print("  ▶ 2. Retrieving Document Chunks (RAG)...")
            retrieved_chunks = []
            if transcript.strip():
                raw_retrieved = agent.rag.index.search(transcript, top_k=3)
                for r in raw_retrieved:
                    c = r["chunk"]
                    retrieved_chunks.append(
                        {
                            "section_id": r["section_id"],
                            "doc_title": c.doc_title,
                            "doc_type": c.doc_type,
                            "status": c.status,
                            "effective_from": c.effective_from,
                            "relevance_score": round(r["composite_score"], 3),
                            "source_file": c.source_file,
                        }
                    )

            # -------------------------------------------------------------
            # 3. Agent Reasoning & Tool Execution (lookup_order / create_ticket)
            # -------------------------------------------------------------
            print("  ▶ 3. Agent Reasoning & Verification...")
            agent_reply = agent.handle_turn(transcript, session)
            print(f"     Agent Text Reply: \"{agent_reply}\"")


        # -------------------------------------------------------------
        # 4. Speech Text Normalization & Text-to-Speech (Gnani TTS)
        # -------------------------------------------------------------
        print("  ▶ 4. Synthesizing Spoken Reply (Gnani TTS)...")
        spoken_wav_name = f"{audio_path.stem}_reply.wav"
        spoken_wav_path = out_dir / spoken_wav_name

        try:
            pipeline.synthesize_speech(
                text=agent_reply,
                output_file=spoken_wav_path,
                voice=voice,
                model=model,
                language=language_code,
            )
            wav_generated = True
            wav_size = spoken_wav_path.stat().st_size
            print(f"     ✓ Spoken Audio File: {spoken_wav_path.name} ({wav_size:,} bytes)")
        except Exception as e:
            print(f"  ❌ TTS Error: {e}")
            wav_generated = False
            wav_size = 0

        # Display retrieved chunks used
        print("  ▶ 5. Document Chunks Used:")
        retrieved_sec_ids = [c_info["section_id"] for c_info in retrieved_chunks] if retrieved_chunks else []
        print(f"     [SECTION-ID RETRIEVED]: {retrieved_sec_ids if retrieved_sec_ids else 'None (Direct tool/verification response)'}")
        if retrieved_chunks:
            for c_info in retrieved_chunks:
                print(
                    f"     • [{c_info['section_id']}] {c_info['doc_title']} "
                    f"({c_info['doc_type']}, status: {c_info['status']}) - Score: {c_info['relevance_score']}"
                )
        else:
            print("     • (Handled via identity verification / order lookup)")

        elapsed = time.time() - file_start
        print(f"  ⏱️  Completed in {elapsed:.2f}s")

        result_record = {
            "input_audio_file": str(audio_path),
            "transcript": transcript,
            "agent_text_reply": agent_reply,
            "spoken_audio_file": str(spoken_wav_path) if wav_generated else None,
            "retrieved_section_ids": retrieved_sec_ids,
            "retrieved_document_chunks": retrieved_chunks,
            "session_state": {
                "order_id": session.order_id,
                "is_verified": session.is_verified,
                "tickets_created": len(session.tickets_created),
                "turn_count": session.turn_count,
            },
        }
        batch_results.append(result_record)

    # Save complete JSON summary report
    summary_path = out_dir / "results_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f"🎉 Batch execution finished! Processed {len(batch_results)} audio files.")
    print(f"📄 Detailed summary report saved to: {summary_path.resolve()}")
    print("=" * 75)

    return batch_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="ZentroMart AI Customer Support Agent - Master Audio Processing Pipeline"
    )
    parser.add_argument(
        "audio_dir",
        nargs="?",
        default=None,
        help="Path to folder containing incoming customer audio files",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        default=None,
        help="Path to folder containing incoming customer audio files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default="output_replies",
        help="Directory to save generated .wav speech replies and summary JSON (default: output_replies/)",
    )
    parser.add_argument(
        "--language",
        default="en-IN",
        help="Language code for STT and TTS (default: en-IN)",
    )
    parser.add_argument(
        "--voice",
        default="Kaveri",
        help="Gnani TTS voice name (default: Kaveri)",
    )
    parser.add_argument(
        "--model",
        default="timbre-v2.5",
        help="Gnani TTS model name (default: timbre-v2.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_folder = args.audio_dir or args.input_dir

    if not input_folder:
        print("Usage: python run.py <audio_folder> [options]")
        print("Example: python run.py sample_audio/ --output-dir output_replies/")
        print("\nCreating a sample demonstration audio for testing...")
        demo_dir = Path("sample_incoming_audio")
        demo_dir.mkdir(exist_ok=True)
        demo_file = demo_dir / "customer_query_sample.wav"

        # Generate a demonstration input audio if none exists
        if not demo_file.exists():
            pipeline = ZentroMartVoicePipeline()
            pipeline.synthesize_speech(
                text="Hello, I need help with an order.",
                output_file=demo_file,
                voice="Kaveri",
            )
            print(f"✓ Generated sample customer audio: {demo_file}")

        input_folder = demo_dir

    try:
        process_audio_folder(
            input_folder=input_folder,
            output_folder=args.output_dir,
            language_code=args.language,
            voice=args.voice,
            model=args.model,
        )
    except Exception as err:
        print(f"\n❌ Error during execution: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
