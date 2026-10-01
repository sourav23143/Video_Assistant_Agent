"""Run the audio-to-analysis pipeline on one YouTube video."""

import os
import traceback

from dotenv import load_dotenv

# Load API keys before importing modules that read environment variables.
load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import (
    generate_title,
    split_transcript as split_summary,
    summarize,
)
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_question,
    split_transcript as split_extraction,
)


SOURCE = "https://www.youtube.com/watch?v=_Q-e_nczWqM&t=223s"
LANGUAGE = "english"  # "english" uses Whisper; "hinglish" uses Sarvam.


def run_stage(name, function):
    """Run one pipeline stage and print its traceback if it fails."""
    try:
        return function()
    except Exception as error:
        print(f"\n{name} failed: {error}")
        traceback.print_exc()
        return None


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Add it to your .env file.")

    if LANGUAGE.lower() == "hinglish" and not os.getenv("SARVAM_API_KEY"):
        raise SystemExit("SARVAM_API_KEY is missing for Hinglish transcription.")

    # Download/convert the source audio and split it into audio files.
    try:
        audio_chunks = process_input(SOURCE)
        if not audio_chunks:
            raise RuntimeError("No audio chunks were created.")

        # Transcribe the audio chunks into one transcript.
        transcript = transcribe_all(audio_chunks, language=LANGUAGE)
        if not transcript.strip():
            raise RuntimeError("Transcription returned an empty transcript.")
    except Exception as error:
        print(f"\nAudio processing or transcription failed: {error}")
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("TRANSCRIPT (first 500 characters)")
    print("=" * 60)
    print(transcript[:500] + ("..." if len(transcript) > 500 else ""))

    # Show how many text chunks the summary and extractors will process.
    print(f"\nSummary chunks: {len(split_summary(transcript))}")
    print(f"Extraction chunks: {len(split_extraction(transcript))}")

    # Generate the title and map-reduce summary.
    title = run_stage("Title generation", lambda: generate_title(transcript))
    summary = run_stage("Summarization", lambda: summarize(transcript))

    # Call each transcript-based extractor separately.
    action_items = run_stage(
        "Action-item extraction",
        lambda: extract_action_items(transcript),
    )
    key_decisions = run_stage(
        "Key-decision extraction",
        lambda: extract_key_decisions(transcript),
    )
    open_questions = run_stage(
        "Open-question extraction",
        lambda: extract_question(transcript),
    )

    if title is not None:
        print("\n" + "=" * 60)
        print(f"TITLE: {title}")

    if summary is not None:
        print("\nSUMMARY")
        print("-" * 60)
        print(summary)

    results = (
        ("ACTION ITEMS", action_items),
        ("KEY DECISIONS", key_decisions),
        ("OPEN QUESTIONS", open_questions),
    )
    for heading, result in results:
        if result is not None:
            print("\n" + "=" * 60)
            print(heading)
            print("=" * 60)
            print(result)


if __name__ == "__main__":
    main()
