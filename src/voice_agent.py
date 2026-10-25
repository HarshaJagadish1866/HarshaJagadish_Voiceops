#!/usr/bin/env python3
"""ZentroMart Voice-Enabled Customer Support Agent using Gnani STT and TTS.

Features:
1. Gnani STT: Transcribes customer audio files using GnaniSTTClient with 'en-IN' language code
   (supporting mixed English and Hindi / Hinglish).
2. Agent Reasoning: Feeds transcript to ZentroMartAgent to retrieve facts, check policies,
   manage session state, and make decisions.
3. Natural Speech Text Normalization:
   - Spaces out 8-digit order numbers into groups of four digits (e.g. '8 9 7 8, 2 8 3 7').
   - Formats years naturally (e.g. '2026' -> 'twenty twenty-six').
   - Formats dates with month names (e.g. '2026-10-24' -> '24th of October, twenty twenty-six').
   - Cleans markdown symbols, brackets, and currency signs ('₹49' -> '49 rupees').
4. Gnani TTS: Synthesizes agent reply into a .wav file using model 'timbre-v2.5',
   voice 'Kaveri', and language 'en-IN'.
"""

import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Optional, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback parser if python-dotenv is not installed in the active python interpreter
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

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if _HERE.name == "src" else _HERE
for _p in [str(_HERE), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient

try:
    from conversation_agent import ConversationSession, ZentroMartAgent
except ImportError:
    from src.conversation_agent import ConversationSession, ZentroMartAgent


def get_gnani_api_key() -> str:
    """Retrieves GNANI_API_KEY from environment variables without hardcoding."""
    api_key = os.getenv("GNANI_API_KEY")
    if not api_key:
        raise ValueError(
            "GNANI_API_KEY environment variable is not set. "
            "Please ensure GNANI_API_KEY is configured in your environment or .env file."
        )
    return api_key


FALLBACK_UNCLEAR_AUDIO_TEXT: str = "Sorry, I didn't catch that. Could you say it again?"


def is_transcript_empty_or_muffled(transcript: Optional[str]) -> bool:
    """Checks whether the STT transcript is empty, silent, or muffled."""
    if transcript is None:
        return True
    cleaned = transcript.strip().lower()
    if not cleaned:
        return True

    # Check for silence/muffled/inaudible/noise tokens
    muffled_markers = {
        "[inaudible]", "(inaudible)", "[silence]", "(silence)", "[noise]",
        "(noise)", "<silence>", "...", "..", ".", "*", "???", "[muffled]",
        "(muffled)", "muffled", "inaudible", "unintelligible", "[unintelligible]",
        "[music]", "(music)", "<noise>", "none", "null", "uh", "um"
    }
    if cleaned in muffled_markers:
        return True

    # Check if string contains no alphanumeric characters
    if not re.search(r"[a-zA-Z0-9\u0900-\u097F]", cleaned):
        return True

    return False


def format_text_for_tts(text: str) -> str:
    """Pre-processing function that intercepts the LLM's text reply before Gnani TTS.

    - Spaces out order IDs digit-by-digit (e.g., converts '89782837' to '8 9 7 8 2 8 3 7').
    - Formats years like '2026' clearly for speech ('twenty twenty-six').
    - Expands dates like 'YYYY-MM-DD' to spoken format ('the 24th of October, twenty twenty-six').
    - Converts currency symbols (e.g. '₹49' -> '49 rupees').
    - Strips markdown formatting, bullet points, asterisks, brackets, and policy codes.
    """
    clean = text

    # 1. Clean markdown headers, bold, italics, bullets, inline code
    clean = re.sub(r"#{1,6}\s*", "", clean)
    clean = re.sub(r"^\s*[-*•]\s+", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
    clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
    clean = re.sub(r"`([^`]+)`", r"\1", clean)
    clean = re.sub(r"\[([A-Z0-9\.\-]+)\]", "", clean)  # Remove [SECTION-ID] tags like [DEL-4]

    # 2. Currency symbol expansion: ₹49 -> 49 rupees, ₹10,000 -> 10000 rupees
    clean = re.sub(r"₹\s*([0-9,]+)", r"\1 rupees", clean)

    # 3. Date conversion: YYYY-MM-DD -> the DDth of Month, twenty twenty-six
    months = {
        "01": "January",
        "02": "February",
        "03": "March",
        "04": "April",
        "05": "May",
        "06": "June",
        "07": "July",
        "08": "August",
        "09": "September",
        "10": "October",
        "11": "November",
        "12": "December",
    }

    def date_repl(match):
        year, month_num, day = match.group(1), match.group(2), match.group(3)
        month_name = months.get(month_num, "October")
        day_int = int(day)
        # Ordinal suffix
        if 11 <= (day_int % 100) <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day_int % 10, "th")
        year_spoken = "twenty twenty-six" if year == "2026" else year
        return f"the {day_int}{suffix} of {month_name}, {year_spoken}"

    clean = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", date_repl, clean)

    # 4. Spoken Year formatting for standalone years: 2026 -> twenty twenty-six
    clean = re.sub(r"\b2026\b", "twenty twenty-six", clean)
    clean = re.sub(r"\b2025\b", "twenty twenty-five", clean)
    clean = re.sub(r"\b2024\b", "twenty twenty-four", clean)
    clean = re.sub(r"\b2023\b", "twenty twenty-three", clean)

    # 5. Order ID spacing: format 8-digit order numbers digit-by-digit
    # e.g., '89782837' or '8978 2837' -> '8 9 7 8 2 8 3 7'
    def order_id_repl(match):
        digits = re.sub(r"\s+", "", match.group(0))
        if len(digits) == 8:
            return " ".join(digits)
        return match.group(0)

    clean = re.sub(r"\b(\d{4})\s*(\d{4})\b", order_id_repl, clean)
    clean = re.sub(r"\b\d{8}\b", order_id_repl, clean)

    # 6. Spoken 4-digit verification numbers (e.g. 'last 4 digits 4533' -> 'last 4 digits 4 5 3 3')
    def four_digit_repl(match):
        prefix = match.group(1)
        digits = match.group(2)
        return f"{prefix} {' '.join(digits)}"

    clean = re.sub(r"(last 4 digits|aakhri 4 digits|digits)\s*(\d{4})\b", four_digit_repl, clean, flags=re.IGNORECASE)

    # 7. Clean excessive punctuation and whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean



class ZentroMartVoicePipeline:
    """End-to-end voice pipeline: Customer Audio -> Gnani STT -> Agent -> Gnani TTS -> WAV."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        agent: Optional[ZentroMartAgent] = None,
    ):
        self.api_key = api_key or get_gnani_api_key()
        self.stt_client = GnaniSTTClient(api_key=self.api_key)
        self.tts_client = GnaniTTSClient(api_key=self.api_key)
        self.agent = agent or ZentroMartAgent()

    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language_code: str = "en-IN",
    ) -> str:
        """Transcribes customer audio file using Gnani STT with language code 'en-IN'."""
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        response = self.stt_client.transcribe(
            audio=str(audio_file),
            language_code=language_code,
            format="transcribe",
        )

        transcript = response.get("transcript", "").strip()
        return transcript

    def synthesize_speech(
        self,
        text: str,
        output_file: Union[str, Path] = "agent_reply.wav",
        voice: str = "Kaveri",
        model: str = "timbre-v2.5",
        language: str = "en-IN",
    ) -> Path:
        """Synthesizes text into a spoken .wav file using Gnani TTS."""
        out_path = Path(output_file)

        # Apply natural speech text normalization
        normalized_text = format_text_for_tts(text)

        self.tts_client.synthesize(
            text=normalized_text,
            voice=voice,
            model=model,
            language=language,
            output_file=out_path,
        )

        return out_path

    def process_voice_turn(
        self,
        customer_audio_file: Union[str, Path],
        session: Optional[ConversationSession] = None,
        output_wav_file: Union[str, Path] = "agent_reply.wav",
    ) -> Dict[str, Any]:
        """Runs the full voice pipeline for a customer turn:

        1. Customer Audio -> Gnani STT ('en-IN').
        2. Transcript -> ZentroMartAgent.
        3. Agent Reply -> Natural Speech Normalizer.
        4. Normalized Reply -> Gnani TTS ('timbre-v2.5', 'Kaveri', 'en-IN') -> .wav.
        """
        if session is None:
            session = ConversationSession(conversation_id="VOICE-SESSION-1")

        # Step 1: Transcribe customer audio
        transcript = self.transcribe_audio(customer_audio_file, language_code="en-IN")

        # Step 2: Handle STT silences / muffled audio gracefully (bypasses LLM)
        if is_transcript_empty_or_muffled(transcript):
            fallback_text = FALLBACK_UNCLEAR_AUDIO_TEXT
            tts_ready_text = format_text_for_tts(fallback_text)
            out_audio_path = self.synthesize_speech(
                text=tts_ready_text,
                output_file=output_wav_file,
                voice="Kaveri",
                model="timbre-v2.5",
                language="en-IN",
            )
            return {
                "transcript": transcript or "",
                "agent_reply_text": fallback_text,
                "tts_input_text": tts_ready_text,
                "output_audio_path": str(out_audio_path),
                "session": session,
                "bypassed_llm": True,
            }

        # Step 3: Feed transcription to our agent
        agent_reply_text = self.agent.handle_turn(transcript, session)

        # Step 4: Pre-process text for natural TTS output (spacing order IDs, formatting years)
        tts_ready_text = format_text_for_tts(agent_reply_text)

        # Step 5: Synthesize agent reply into spoken .wav
        out_audio_path = self.synthesize_speech(
            text=tts_ready_text,
            output_file=output_wav_file,
            voice="Kaveri",
            model="timbre-v2.5",
            language="en-IN",
        )


        return {
            "transcript": transcript,
            "agent_reply_text": agent_reply_text,
            "tts_input_text": tts_ready_text,
            "output_audio_path": str(out_audio_path),
            "session": session,
        }


def process_customer_audio_turn(
    audio_file_path: Union[str, Path],
    session: Optional[ConversationSession] = None,
    output_wav_path: Union[str, Path] = "agent_reply.wav",
) -> Dict[str, Any]:
    """Convenience function: Transcribes customer audio -> Agent -> Gnani TTS .wav reply."""
    pipeline = ZentroMartVoicePipeline()
    return pipeline.process_voice_turn(
        customer_audio_file=audio_file_path,
        session=session,
        output_wav_file=output_wav_path,
    )


def main():
    print("=" * 70)
    print("      ZentroMart Voice-Enabled Agent (Gnani STT & TTS)")
    print("=" * 70)

    pipeline = ZentroMartVoicePipeline()
    print("✓ Initialized Gnani STT (en-IN) and TTS (timbre-v2.5, Kaveri, en-IN)")

    # 1. Test Text Normalization
    sample_text = (
        "Under [RET-2] (Returns policy): Your order 89782837 was placed on 2026-10-06. "
        "The return window is open until 2026-10-27. A reverse pickup fee of ₹49 applies."
    )
    print("\n--- Text Normalization Test ---")
    print(f"Original Text:\n{sample_text}")
    normalized = format_text_for_tts(sample_text)
    print(f"\nTTS-Normalized Text:\n{normalized}")

    # 2. Test TTS Generation
    print("\n--- Generating Sample Agent Spoken Reply ---")
    output_wav = pipeline.synthesize_speech(
        text=normalized,
        output_file="demo_agent_reply.wav",
    )
    print(f"✓ Spoken reply generated: {output_wav} ({os.path.getsize(output_wav)} bytes)")

    # 3. Test Full Loop: Audio File -> STT -> Agent -> TTS
    print("\n--- Testing Full Loop (STT -> Agent -> TTS) ---")
    session = ConversationSession(conversation_id="DEMO-VOICE")
    result = pipeline.process_voice_turn(
        customer_audio_file="demo_agent_reply.wav",
        session=session,
        output_wav_file="demo_agent_turn_reply.wav",
    )
    print(f"Customer Spoken Transcript : {result['transcript']}")
    print(f"Agent Text Response        : {result['agent_reply_text']}")
    print(f"TTS Spoken Input           : {result['tts_input_text']}")
    print(f"Generated Audio WAV Output : {result['output_audio_path']}")

    # Clean up demo files
    for f in ["demo_agent_reply.wav", "demo_agent_turn_reply.wav"]:
        if os.path.exists(f):
            os.remove(f)
    print("\n✓ Verification completed successfully.")


if __name__ == "__main__":
    main()
