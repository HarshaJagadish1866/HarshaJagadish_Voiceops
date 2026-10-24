#!/usr/bin/env bash
# =====================================================================
# ZentroMart AI Customer Support Agent - Master Run Shell Script
# =====================================================================

set -e

AUDIO_DIR="$1"

if [ -z "$AUDIO_DIR" ]; then
    echo "Usage: ./run.sh <path_to_audio_folder> [optional_output_folder]"
    echo "Example: ./run.sh sample_incoming_audio/ output_replies/"
    echo ""
    echo "Running with default demonstration audio folder..."
    python3 run.py
else
    OUTPUT_DIR="${2:-output_replies}"
    python3 run.py "$AUDIO_DIR" --output-dir "$OUTPUT_DIR"
fi
