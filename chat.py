#!/usr/bin/env python3
"""Interactive Terminal Chat with ZentroMart AI Customer Support Agent.

Run this script to talk directly to the agent in real time.
Features:
- Multi-turn conversation state memory (remembers order ID, problem, identity).
- Identity Gatekeeper: Asks for last 4 digits of phone or pincode before revealing order details.
- Precedence RAG: Answers from ZentroMart knowledge base (policies, returns, delivery, warranties).
- Audio speech output (optional): Speaks agent replies aloud via Gnani TTS and macOS afplay!
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Silence Gnani SDK internal loggers completely
gnani_log = logging.getLogger("gnani")
gnani_log.setLevel(logging.CRITICAL)
gnani_log.disabled = True
for h in gnani_log.handlers:
    gnani_log.removeHandler(h)

# Load environment fresh
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from conversation_agent import ConversationSession, ZentroMartAgent
from voice_agent import format_text_for_tts, get_gnani_api_key

def main():
    print("=" * 70)
    print("      🎙️  ZentroMart AI Customer Support Agent - Live Chat")
    print("=" * 70)
    print("• Type your message in English or Hinglish.")
    print("• Type 'voice on' or 'voice off' to toggle spoken voice replies.")
    print("• Type 'reset' to start a new conversation session.")
    print("• Type 'exit' or 'quit' to end the chat.")
    print("=" * 70 + "\n")

    agent = ZentroMartAgent()
    session = ConversationSession(conversation_id="LIVE-USER-SESSION")
    
    # Check if Gnani TTS voice is available
    voice_enabled = False
    tts_client = None
    api_key = os.getenv("GNANI_API_KEY", "").strip()

    if api_key:
        try:
            from gnani.tts import GnaniTTSClient
            # Silence Gnani SDK internal loggers completely
            gnani_log = logging.getLogger("gnani")
            gnani_log.handlers.clear()
            gnani_log.setLevel(logging.CRITICAL)
            gnani_log.disabled = True
            gnani_log.propagate = False

            tts_client = GnaniTTSClient(api_key=api_key)
            voice_enabled = True
            print("🔊 Gnani Voice TTS: ENABLED (Agent will speak replies aloud)")
        except Exception as e:
            print(f"⚠️  Voice TTS initialized in text-only mode ({e})")
    else:
        print("ℹ️  GNANI_API_KEY is empty in .env. Running in Text Mode.")
        print("   (To hear spoken audio replies, paste your key into .env and restart)\n")

    turn_count = 0

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! Have a great day.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit", "q"):
            print("Goodbye! Thanks for visiting ZentroMart.")
            break
        elif cmd == "reset":
            session = ConversationSession(conversation_id=f"LIVE-USER-SESSION-{turn_count}")
            turn_count = 0
            print("🔄 Session reset. You are now starting a fresh conversation.")
            continue
        elif cmd == "voice on":
            if not api_key:
                print("❌ Cannot enable voice: GNANI_API_KEY is empty in .env.")
            else:
                voice_enabled = True
                print("🔊 Voice replies turned ON.")
            continue
        elif cmd == "voice off":
            voice_enabled = False
            print("🔇 Voice replies turned OFF.")
            continue

        turn_count += 1

        # Process turn through agent
        raw_agent_reply = agent.handle_turn(user_input, session)
        spoken_text = format_text_for_tts(raw_agent_reply)
        print(f"\nAgent: {spoken_text}")

        # Spoken audio output
        if voice_enabled:
            temp_audio = "/tmp/agent_live_reply.wav"
            spoken_success = False

            if tts_client:
                try:
                    tts_client.synthesize(
                        text=spoken_text,
                        output_file=temp_audio,
                        model="timbre-v2.5",
                        voice="Kaveri",
                        language="en-IN",
                    )
                    if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 100:
                        subprocess.run(["afplay", temp_audio], check=False)
                        spoken_success = True
                except Exception:
                    pass

            # Fallback to macOS native speech if Gnani API key is not active
            if not spoken_success:
                try:
                    subprocess.run(["say", spoken_text], check=False)
                except Exception:
                    pass

if __name__ == "__main__":
    main()
