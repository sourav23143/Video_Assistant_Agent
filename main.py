from dotenv import load_dotenv
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_question


load_dotenv()


def extract_transcript_insights(transcript: str) -> dict[str, str]:
    """Extract the three insight categories for use by the application."""
    return {
        "action_items": extract_action_items(transcript),
        "key_decisions": extract_key_decisions(transcript),
        "open_question": extract_question(transcript),
    }
