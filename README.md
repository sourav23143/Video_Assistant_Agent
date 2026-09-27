<div align="center">

# 🎬 AI Video Assistant

### *Transcribe · Summarise · Chat with your Meetings*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Mistral AI](https://img.shields.io/badge/Mistral_AI-Small-FF7000?style=for-the-badge)](https://mistral.ai)
[![Whisper](https://img.shields.io/badge/OpenAI_Whisper-Local-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

An intelligent, end-to-end AI pipeline that takes any **YouTube video** or **local audio/video file**, transcribes it, generates professional meeting summaries, extracts actionable insights, and lets you **chat with your meeting** using a RAG-powered Q&A engine — all through a stunning Streamlit UI.

<br/>

</div>

---

## 📑 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How It Works — Deep Dive](#-how-it-works--deep-dive)
  - [Audio Processing Pipeline](#1-audio-processing-pipeline)
  - [Transcription Engine](#2-transcription-engine)
  - [LLM-Powered Analysis](#3-llm-powered-analysis)
  - [RAG Chat Engine](#4-rag-chat-engine)
  - [Streamlit Web Interface](#5-streamlit-web-interface)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the App](#running-the-app)
- [Usage Guide](#-usage-guide)
- [API Keys Required](#-api-keys-required)
- [Environment Variables](#-environment-variables)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [Credits](#-credits)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **YouTube Integration** | Paste any YouTube URL — audio is automatically downloaded and processed |
| 📁 **Local File Support** | Supports local `.mp4`, `.mp3`, `.wav`, `.webm`, and other audio/video formats |
| 🗣️ **Dual Transcription Engines** | **English** → OpenAI Whisper (runs locally) · **Hinglish** → Sarvam AI (cloud API with auto-translation) |
| 📋 **Smart Summarisation** | Map-Reduce summarisation using Mistral AI for long transcripts with chunked processing |
| ✅ **Action Item Extraction** | Automatically identifies tasks, owners, and deadlines from meetings |
| 🔑 **Key Decision Detection** | Extracts all important decisions made during the meeting |
| ❓ **Open Question Tracking** | Finds unresolved questions and topics needing follow-up |
| 🏷️ **Auto Title Generation** | Generates a concise, professional meeting title from transcript content |
| 💬 **RAG-Powered Chat** | Ask natural-language questions about your meeting — answers grounded in transcript context |
| 🎨 **Premium Dark UI** | Sleek, modern Streamlit interface with glassmorphism, gradient accents, and micro-animations |
| 📊 **Live Pipeline Status** | Real-time sidebar progress tracker showing each pipeline stage |
| 🖥️ **CLI Mode** | Full pipeline accessible from the command line via `main.py` |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INPUT                                 │
│              YouTube URL  ──or──  Local File Path               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  AUDIO PROCESSING LAYER                         │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐     │
│  │  yt-dlp      │   │  pydub       │   │  FFmpeg          │     │
│  │  (download)  │   │  (convert)   │   │  (extract audio) │     │
│  └──────┬───────┘   └──────┬───────┘   └──────────────────┘     │
│         └──────────┬───────┘                                    │
│                    ▼                                            │
│         ┌──────────────────┐                                    │
│         │  chunk_audio()   │  → 10-minute WAV chunks            │
│         └──────────────────┘                                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TRANSCRIPTION LAYER                             │
│  ┌──────────────────────┐  ┌──────────────────────────────┐     │
│  │  OpenAI Whisper       │  │  Sarvam AI API               │     │
│  │  (English, local)     │  │  (Hinglish → English,cloud)  │     │
│  │  Model: small         │  │  25s sub-chunks              │     │
│  └──────────────────────┘  └──────────────────────────────┘     │
└───────────────────────────┬─────────────────────────────────────┘
                            │  Full transcript text
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LLM ANALYSIS LAYER (Mistral AI)                │
│  ┌────────────┐ ┌──────────────┐ ┌───────────┐ ┌────────────┐  │
│  │  Title     │ │  Summary     │ │  Action   │ │  Decisions │  │
│  │  Generator │ │  (Map-Reduce)│ │  Items    │ │  & Q's     │  │
│  └────────────┘ └──────────────┘ └───────────┘ └────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RAG CHAT ENGINE                                 │
│  ┌──────────────┐  ┌────────────┐  ┌────────────────────┐      │
│  │  Text Splitter│  │  ChromaDB  │  │  MiniLM Embeddings │      │
│  │  (500 chars)  │  │  (vector   │  │  (HuggingFace)     │      │
│  │              │  │   store)   │  │                    │      │
│  └──────┬───────┘  └─────┬──────┘  └────────┬───────────┘      │
│         └────────┬───────┘                   │                  │
│                  ▼                           │                  │
│         ┌──────────────────┐                 │                  │
│         │  LCEL RAG Chain  │◄────────────────┘                  │
│         │  (Retriever +    │                                    │
│         │   Mistral LLM)   │                                    │
│         └──────────────────┘                                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                              │
│  ┌─────────────────────┐    ┌─────────────────────────┐         │
│  │  Streamlit Web UI   │    │  CLI (main.py)          │         │
│  │  (app.py)           │    │  Interactive terminal    │         │
│  └─────────────────────┘    └─────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Audio Acquisition** | `yt-dlp` | Download audio from YouTube URLs |
| **Audio Processing** | `pydub`, `FFmpeg` | Convert formats, chunk audio into segments |
| **Speech-to-Text (English)** | `OpenAI Whisper` | Local model transcription (configurable size) |
| **Speech-to-Text (Hinglish)** | `Sarvam AI API` | Cloud-based Hindi/Hinglish STT with auto English translation |
| **LLM (Analysis)** | `Mistral AI` (`mistral-small-latest`) | Summarisation, extraction, title gen, RAG answers |
| **LLM Orchestration** | `LangChain LCEL` | Prompt chaining, map-reduce, runnable pipelines |
| **Vector Store** | `ChromaDB` | Local persistent vector database for RAG |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) | Sentence embeddings for semantic search |
| **Frontend** | `Streamlit` | Interactive web UI with custom dark theme |
| **Environment** | `python-dotenv` | Secure API key management via `.env` files |

---

## 📂 Project Structure

```
AI_Video_Assistant/
│
├── app.py                      # Streamlit web application (main UI)
├── main.py                     # CLI entry point for terminal usage
├── test.py                     # Quick-test script for pipeline validation
├── requirements.txt            # Python dependencies with version pins
├── .env.example                # Template for environment variables
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
├── core/                       # Core processing modules
│   ├── __init__.py
│   ├── transcriber.py          # Whisper & Sarvam transcription engines
│   ├── summarizer.py           # Map-Reduce summarisation + title generation
│   ├── extractor.py            # Action items, decisions, questions extraction
│   ├── rag_engine.py           # RAG chain construction & Q&A interface
│   └── vector_store.py         # ChromaDB vector store management
│
└── utils/                      # Utility modules
    ├── __init__.py
    └── audio_processor.py      # YouTube download, format conversion, chunking
```

---

## 🔬 How It Works — Deep Dive

### 1. Audio Processing Pipeline

**File:** `utils/audio_processor.py`

The pipeline begins by acquiring and preparing audio:

1. **Input Detection** — The `process_input()` function auto-detects whether the source is a YouTube URL or a local file path.

2. **YouTube Download** — For URLs, `yt-dlp` downloads the best available audio stream and uses FFmpeg to convert it to WAV format (saved to the `downloades/` directory).

3. **Format Conversion** — For local files, `pydub` converts any audio/video format to a standardised **mono 16kHz WAV** file, which is the optimal format for speech recognition.

4. **Audio Chunking** — Long recordings are split into **10-minute chunks** to stay within model context limits and manage memory efficiently. Each chunk is exported as a separate WAV file.

```python
# Chunking logic (simplified)
chunk_ms = 10 * 60 * 1000  # 10 minutes in milliseconds
for i, start in enumerate(range(0, len(audio), chunk_ms)):
    chunk = audio[start : start + chunk_ms]
    chunk.export(f"chunk_{i}.wav", format="wav")
```

---

### 2. Transcription Engine

**File:** `core/transcriber.py`

The system supports two transcription backends selected by the user's language choice:

#### English Mode → OpenAI Whisper (Local)
- Loads the Whisper model **once** (lazy-loaded singleton) to avoid repeated loading overhead.
- Model size is configurable via the `WHISPER_MODEL` environment variable (`tiny`, `base`, `small`, `medium`, `large`).
- Runs entirely on your local machine — no API calls needed.

#### Hinglish Mode → Sarvam AI (Cloud)
- Sends audio to **Sarvam AI's speech-to-text-translate API** which handles Hindi/Hinglish audio and returns English transcripts.
- Sarvam's sync API has a **30-second limit** per request, so each 10-minute chunk is further sliced into **25-second sub-pieces** (with a 5-second safety margin).
- Sub-piece transcripts are concatenated to form the full chunk transcript.
- Temporary sub-piece WAV files are cleaned up after processing.

```python
# Language routing
if language == "hinglish":
    return transcribe_chunk_sarvam(chunk_path)   # Cloud API
return transcribe_chunk_whisper(chunk_path)       # Local model
```

---

### 3. LLM-Powered Analysis

**Files:** `core/summarizer.py`, `core/extractor.py`

All analysis is powered by **Mistral AI** (`mistral-small-latest`) through **LangChain LCEL** (LangChain Expression Language) chains:

#### Title Generation (`summarizer.py`)
- Feeds the first 2,000 characters of the transcript to an LLM chain.
- System prompt constrains output to a **max 8-word professional title**.

#### Map-Reduce Summarisation (`summarizer.py`)
- **Map Phase:** The transcript is split into 3,000-character chunks (with 200-char overlap). Each chunk is independently summarised.
- **Reduce Phase:** All chunk summaries are combined and fed to a final LLM call that produces a unified, professional bullet-point summary.

```
Transcript → [Chunk₁, Chunk₂, ..., Chunkₙ]
                ↓         ↓              ↓
           [Summary₁, Summary₂, ..., Summaryₙ]  ← Map Phase
                ↓         ↓              ↓
              Combined → Final Summary              ← Reduce Phase
```

#### Extraction Chains (`extractor.py`)
Three specialised LLM chains extract structured information:

| Chain | Output |
|-------|--------|
| `extract_action_items()` | Numbered list of tasks with owner + deadline |
| `extract_key_decisions()` | Numbered list of decisions made |
| `extract_questions()` | Numbered list of unresolved questions |

Each chain uses `build_chain()` — a reusable factory that constructs a `RunnablePassthrough → Prompt → LLM → StrOutputParser` LCEL pipeline.

---

### 4. RAG Chat Engine

**Files:** `core/rag_engine.py`, `core/vector_store.py`

After analysis, the system builds a **Retrieval-Augmented Generation (RAG)** pipeline so users can ask natural-language questions about the meeting:

#### Vector Store Construction (`vector_store.py`)
1. The full transcript is split into **500-character chunks** (50-char overlap) using `RecursiveCharacterTextSplitter`.
2. Each chunk is wrapped in a LangChain `Document` with metadata (chunk index).
3. **Sentence embeddings** are generated using `all-MiniLM-L6-v2` from HuggingFace (runs on CPU).
4. Embeddings are stored in a local **ChromaDB** vector database (persisted to `vector_db/`).

#### RAG Chain (`rag_engine.py`)
The LCEL RAG pipeline works as follows:

```
User Question
      │
      ├──→ Retriever (top-4 similar chunks from ChromaDB)
      │         │
      │         ▼
      │    format_docs() → concatenated context
      │         │
      └────────►├──→ Prompt Template (system + context + question)
                │
                ▼
           Mistral LLM → StrOutputParser → Answer
```

- The retriever uses **cosine similarity** search to find the 4 most relevant transcript chunks.
- The LLM is instructed to answer **only from the provided context** — if the information isn't in the transcript, it will say so.
- Chat history is maintained in Streamlit's session state.

---

### 5. Streamlit Web Interface

**File:** `app.py`

The web UI is built with Streamlit and features:

- **Custom Dark Theme** — CSS variables define a cohesive dark color palette (`#0a0a0f` background, purple/cyan accents) with JetBrains Mono and Syne fonts.
- **Animated Grid Background** — Subtle CSS grid pattern with purple-tinted lines.
- **Sidebar Controls** — Input URL/path, language selection, and analyse button.
- **Live Pipeline Tracker** — Real-time status dots (pending → active → done) for each processing stage.
- **Card-Based Results** — Summary, action items, decisions, and questions displayed in styled cards with gradient accent borders.
- **Chat Interface** — Custom-styled chat bubbles with user/assistant message differentiation.
- **Expandable Transcript** — Full transcript available in a collapsible expander widget.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **FFmpeg** installed and available in PATH
  - Windows: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- **Mistral AI API Key** (required)
- **Sarvam AI API Key** (optional — only for Hinglish mode)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/sourav23143/AI_Video_Assistant.git
cd AI_Video_Assistant

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# MISTRAL_API_KEY=your_key_here        (Required)
# SARVAM_API_KEY=your_key_here         (Optional — Hinglish only)
# WHISPER_MODEL=small                   (Optional — tiny/base/small/medium/large)
```

### Running the App

#### Option 1: Streamlit Web UI (Recommended)
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

#### Option 2: Command-Line Interface
```bash
python main.py
```
Follow the prompts to enter a YouTube URL or file path, then chat with your meeting in the terminal.

### ☁️ Deployment (Streamlit Community Cloud)

This app is fully configured for free deployment on [Streamlit Community Cloud](https://share.streamlit.io/).

1. Push your repository to GitHub.
2. Log in to Streamlit Community Cloud and click **New app**.
3. Select your repository, branch, and set the main file path to `app.py`.
4. Click **Advanced settings** and set the **Python version to 3.11 or 3.12** (Important: 3.13+ is not recommended due to dependencies).
5. Paste your API keys into the **Secrets** block:
   ```toml
   MISTRAL_API_KEY = "your-mistral-key"
   SARVAM_API_KEY = "your-sarvam-key"
   WHISPER_MODEL = "tiny"
   ```
6. Click **Deploy!**

*Note: The included `packages.txt` automatically installs `ffmpeg`, and `requirements.txt` is pre-configured to use the CPU-only version of PyTorch to stay within the 1GB RAM free tier limit.*

---

## 📖 Usage Guide

1. **Launch** the Streamlit app with `streamlit run app.py`
2. **Enter** a YouTube URL (e.g., `https://youtube.com/watch?v=...`) or a local file path in the sidebar
3. **Select** the language:
   - `english` — Uses local Whisper model (no API needed for transcription)
   - `hinglish` — Uses Sarvam AI cloud API (requires `SARVAM_API_KEY`)
4. **Click** the ⚡ **Analyse** button
5. **Watch** the pipeline progress in the sidebar (Audio → Transcript → Title → Summary → Extraction → RAG)
6. **Review** the results: title, summary, action items, key decisions, and open questions
7. **Chat** with your meeting using the Q&A box at the bottom — ask anything about the transcript

---

## 🔑 API Keys Required

| Key | Required | Free Tier | Get It At |
|-----|----------|-----------|-----------|
| `MISTRAL_API_KEY` | ✅ Yes | ✅ Yes | [console.mistral.ai](https://console.mistral.ai/) |
| `SARVAM_API_KEY` | ❌ Only for Hinglish | ✅ Yes | [sarvam.ai](https://www.sarvam.ai/) |

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | — | Your Mistral AI API key |
| `SARVAM_API_KEY` | — | Your Sarvam AI API key (Hinglish mode only) |
| `WHISPER_MODEL` | `small` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `SARVAM_STT_MODEL` | `saaras:v2.5` | Sarvam speech-to-text model version |

### Whisper Model Size Guide

| Model | Parameters | English-Only | RAM Required | Relative Speed |
|-------|-----------|-------------|-------------|----------------|
| `tiny` | 39M | ✅ | ~1 GB | ~32x |
| `base` | 74M | ✅ | ~1 GB | ~16x |
| `small` | 244M | ✅ | ~2 GB | ~6x |
| `medium` | 769M | ✅ | ~5 GB | ~2x |
| `large` | 1550M | ✅ | ~10 GB | 1x |

---

## 🖼️ Screenshots

> Launch the app with `streamlit run app.py` to see the stunning dark-themed UI with:
> - Animated grid background
> - Gradient accent cards
> - Real-time pipeline status tracker
> - Interactive chat interface

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Ideas for Contributions
- 📄 PDF/TXT export for meeting reports
- 🌍 Additional language support
- 🔊 Speaker diarisation (who said what)
- 📊 Meeting analytics dashboard
- 🔗 Calendar integration
- 🧪 Unit tests with pytest

---

## 🙏 Built With

- **OpenAI Whisper** — Local speech-to-text model
- **Mistral AI** — LLM for analysis and RAG
- **Sarvam AI** — Hinglish speech-to-text translation
- **LangChain** — LLM orchestration framework
- **ChromaDB** — Vector database
- **HuggingFace** — Sentence embedding models
- **Streamlit** — Web application framework

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️**

*If you found this helpful, please ⭐ star the repository!*

</div>
