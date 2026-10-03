# VidSage

**Turn video into clear notes, practical takeaways, and transcript answers.**

VidSage is a Streamlit app that analyzes YouTube videos and local audio/video files. It transcribes the audio, creates a summary and title, extracts action items and decisions, identifies open questions, and lets you ask questions grounded in the transcript.

## Features

- Analyze a YouTube URL or a local audio/video file.
- Transcribe English audio locally with OpenAI Whisper.
- Transcribe Hinglish audio with Sarvam and return an English transcript.
- Generate a title and a map-reduce summary with OpenAI `gpt-4o-mini`.
- Extract action items, key decisions, and open questions.
- Ask follow-up questions using transcript retrieval with ChromaDB and Hugging Face embeddings.
- Review results in the Streamlit UI or run the pipeline from the terminal.

## How it works

1. `yt-dlp` downloads audio from a YouTube URL; local media is read with FFmpeg through pydub.
2. YouTube audio is extracted to WAV. Local files are converted to mono, 16 kHz WAV. Both are split into 10-minute chunks.
3. Whisper transcribes English audio locally. Sarvam processes Hinglish audio in 25-second pieces and translates it to English.
4. OpenAI generates a title and summary, then extracts action items, decisions, and unanswered questions.
5. Transcript chunks are embedded and stored in a local ChromaDB database for retrieval-based chat.

## Requirements

- Python 3.10 or newer.
- FFmpeg installed and available on `PATH`. Check with `ffmpeg -version` and `ffprobe -version`.
- An OpenAI API key for title generation, summarization, extraction, and transcript chat.
- A Sarvam API key when using Hinglish transcription.
- Internet access for YouTube downloads, API calls, and the first download of the Whisper and embedding models.

## Installation

From PowerShell in the project directory:

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

If you do not use `uv`, create a virtual environment and install the same requirements with pip:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root, next to `app.py`. Add your own keys; do not commit this file.

```dotenv
OPENAI_API_KEY=your_openai_api_key
SARVAM_API_KEY=your_sarvam_api_key
WHISPER_MODEL=small
SARVAM_STT_MODEL=saaras:v2.5
YTDLP_COOKIES_FROM_BROWSER=firefox
YTDLP_COOKIES_FILE=
```

`OPENAI_API_KEY` is required for analysis and chat. `SARVAM_API_KEY` is required only for Hinglish. `WHISPER_MODEL` defaults to `small`; `SARVAM_STT_MODEL` defaults to `saaras:v2.5`.

`YTDLP_COOKIES_FROM_BROWSER` selects the browser yt-dlp reads when no cookie file is configured. The code defaults to Chrome; set it to the browser where you are signed in, such as `firefox` or `edge`. Alternatively, set `YTDLP_COOKIES_FILE` to a Netscape-format `cookies.txt` file. When set, the cookie file takes precedence over browser cookies. Cookie files contain session credentials: keep them private and do not commit them.

## Run the app

With the virtual environment activated:

```powershell
streamlit run app.py
```

Open the Local URL printed by Streamlit, usually `http://localhost:8501`. Enter a YouTube URL or local media path, choose the language, and select **Analyze video**. The results appear in tabs, with transcript chat below them.

The repository also includes an interactive terminal runner:

```powershell
python main.py
```

It prompts for a source and language, prints the analysis, then accepts transcript questions until you type `exit`.

## Project layout

```text
app.py                  Streamlit interface
main.py                 Interactive terminal runner
core/
  transcriber.py        Whisper and Sarvam transcription
  summarizer.py         Title generation and map-reduce summary
  extractor.py          Action items, decisions, and open questions
  vector_store.py       ChromaDB and Hugging Face embeddings
  rag_engine.py         Transcript retrieval and question answering
utils/
  audio_processor.py    YouTube/local audio, conversion, and chunking
.streamlit/
  config.toml           Streamlit theme and server settings
requirements.txt        Python dependencies
downloads/              Generated audio and chunks
vector_db/              Local ChromaDB data
```

## Troubleshooting

- **FFmpeg is missing:** install FFmpeg and make sure `ffmpeg` and `ffprobe` are on `PATH`, then reopen the terminal.
- **YouTube asks you to sign in or blocks the download:** check that the video is accessible and configure a valid browser cookie source or cookie file. YouTube may invalidate cookies; never share or publish them.
- **Sarvam reports a missing key:** add `SARVAM_API_KEY` to `.env` when Hinglish is selected.
- **The browser page is blank:** leave the Streamlit terminal running and try `http://127.0.0.1:8501`.
- **Windows logs `WinError 10054`:** this can occur when the browser connection closes or reconnects. If the page loads and analysis continues, it is not a pipeline failure.
- **Hugging Face embedding deprecation warning:** the current embedding class still runs, but LangChain recommends `HuggingFaceEmbeddings` from `langchain_huggingface` for new code.

## Data and credentials

Audio downloads and chunks are written under `downloads/`. ChromaDB stores transcript chunks and embeddings under `vector_db/`. The repository ignores `downloads/`, `.env`, and `cookies.txt`; it does not currently ignore `vector_db/`, so treat that directory as potentially sensitive and do not commit it. Keep exported YouTube cookies private .

____
