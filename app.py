"""VidSage: turn videos into useful notes and transcript answers."""

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="VidSage | Video insights",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_question
from core.rag_engine import build_rag_chain, ask_question


for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def run_step(status, key: str, label: str, function):
    """Show a pipeline step and keep its status for the sidebar."""
    st.session_state.pipeline_steps[key] = "active"
    status.write(label)
    result = function()
    st.session_state.pipeline_steps[key] = "done"
    return result


with st.sidebar:
    st.markdown("### :material/video_library: VidSage")
    st.caption("Video intelligence")
    st.divider()
    st.markdown("**Your video, organized**")
    st.caption("Summaries, action items, key decisions, open questions, and transcript chat.")

    if st.session_state.pipeline_done:
        st.divider()
        st.success("Analysis ready", icon=":material/check_circle:")
        st.caption("Your results and transcript chat are available on this page.")


st.title("Analyze a video", icon=":material/video_library:")
st.caption("Paste a YouTube link or enter a local media path. VidSage will turn it into clear notes and transcript answers.")

with st.container(border=True):
    st.subheader("Video source")
    with st.form("analysis_form", clear_on_submit=False, border=False):
        source = st.text_input(
            "YouTube URL or local media path",
            placeholder="Paste a YouTube link or enter a video or audio file path",
        )
        language_column, action_column = st.columns([2, 1], vertical_alignment="bottom")
        with language_column:
            language = st.selectbox(
                "Language",
                options=["english", "hinglish"],
                format_func=lambda value: "English" if value == "english" else "Hinglish",
            )
        with action_column:
            run_btn = st.form_submit_button(
                "Analyze video",
                type="primary",
                icon=":material/auto_awesome:",
                width="stretch",
            )


if run_btn:
    if not source.strip():
        st.warning("Enter a YouTube URL or local media file path in the video source form above.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        status = None
        try:
            with st.status("Analyzing your video", expanded=True) as status:
                chunks = run_step(
                    status,
                    "audio",
                    "Preparing audio from the video…",
                    lambda: process_input(source.strip()),
                )
                if not chunks:
                    raise RuntimeError("No audio chunks were created from this source.")

                transcript = run_step(
                    status,
                    "transcript",
                    "Transcribing the audio…",
                    lambda: transcribe_all(chunks, language),
                )
                if not transcript or not transcript.strip():
                    raise RuntimeError("The transcription was empty. Check the source and try again.")

                title = run_step(
                    status,
                    "title",
                    "Creating a title…",
                    lambda: generate_title(transcript),
                )
                summary = run_step(
                    status,
                    "summary",
                    "Writing the summary…",
                    lambda: summarize(transcript),
                )

                st.session_state.pipeline_steps["extract"] = "active"
                status.write("Finding action items, decisions, and open questions…")
                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_question(transcript)
                st.session_state.pipeline_steps["extract"] = "done"

                rag_chain = run_step(
                    status,
                    "rag",
                    "Preparing transcript chat…",
                    lambda: build_rag_chain(transcript),
                )

                st.session_state.result = {
                    "title": title,
                    "transcript": transcript,
                    "summary": summary,
                    "action_items": action_items,
                    "key_decisions": decisions,
                    "open_questions": questions,
                    "rag_chain": rag_chain,
                }
                st.session_state.pipeline_done = True
                status.update(
                    label="Your video analysis is ready",
                    state="complete",
                    expanded=False,
                )
        except Exception as error:
            if status is not None:
                status.update(label="Analysis stopped", state="error", expanded=True)
            for key, state in st.session_state.pipeline_steps.items():
                if state == "active":
                    st.session_state.pipeline_steps[key] = "pending"
            st.error("VidSage couldn’t finish analyzing this video. Check the source and try again.")
            with st.expander("Error details"):
                st.exception(error)


if st.session_state.result:
    result = st.session_state.result
    st.title(result["title"], icon=":material/description:")
    st.caption("Analysis complete · Explore the notes below or ask a question about the video.")

    overview_tab, actions_tab, decisions_tab, transcript_tab = st.tabs(
        ["Overview", "Action items", "Decisions & questions", "Transcript"]
    )

    with overview_tab:
        st.subheader("Summary")
        st.markdown(result["summary"] or "No summary was generated.")

    with actions_tab:
        st.subheader("Action items")
        st.markdown(result["action_items"] or "No action items were found.")

    with decisions_tab:
        st.subheader("Key decisions")
        st.markdown(result["key_decisions"] or "No key decisions were found.")
        st.subheader("Open questions")
        st.markdown(result["open_questions"] or "No open questions were found.")

    with transcript_tab:
        st.text_area(
            "Full transcript",
            value=result["transcript"],
            height=420,
            disabled=True,
            help="Scroll through or copy the transcript text.",
        )

    heading_column, control_column = st.columns([5, 1], vertical_alignment="center")
    with heading_column:
        st.header("Ask VidSage", icon=":material/chat:")
    with control_column:
        if st.session_state.chat_history and st.button(
            "Clear chat",
            type="secondary",
            icon=":material/delete_sweep:",
            width="content",
        ):
            st.session_state.chat_history = []
            st.rerun()

    st.caption("Answers are based on this video's transcript.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.chat_history:
        st.caption("Try asking what the main takeaways were, what actions were suggested, or which questions remain open.")

    prompt = st.chat_input("Ask a question about this video")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=":material/video_library:"):
            with st.spinner("Searching the transcript…"):
                try:
                    answer = ask_question(result["rag_chain"], prompt)
                    st.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as error:
                    st.error("I couldn’t answer that just now. Please try again.")
                    with st.expander("Error details"):
                        st.exception(error)

else:
    st.subheader("Start with a video")
    st.write("Add a YouTube link or local media path in the video source form above, choose the language, and select **Analyze video**.")

    st.subheader("What you’ll get")
    feature_columns = st.columns(3)
    features = [
        (":material/summarize:", "Summary", "See the main ideas at a glance."),
        (":material/checklist:", "Action items", "Capture practical next steps."),
        (":material/forum:", "Transcript chat", "Ask questions about what was said."),
    ]
    for column, (icon, title, description) in zip(feature_columns, features):
        with column:
            with st.container(border=True):
                st.markdown(f"### {icon} {title}")
                st.caption(description)
