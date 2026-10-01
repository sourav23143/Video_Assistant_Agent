import yt_dlp
from pydub import AudioSegment #we will use this when we will do chunking
import os

DOWNLOAD_DIR = 'downloads'  #this will be the directory where all the things will bw saved
os.makedirs(DOWNLOAD_DIR, exist_ok= True) #this will create a directory if it is not existing


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "force_ipv4": True,
        "sleep_interval_requests": 1,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True, #this will supress all the download progress log in the terminal, remove this if we wnat to see log while testing
    }

    # Prefer an explicit cookie file when configured. Otherwise, read cookies
    # from the selected browser (Chrome by default).
    cookie_file = os.getenv("YTDLP_COOKIES_FILE", "").strip()
    cookie_file = os.path.expandvars(os.path.expanduser(cookie_file)) if cookie_file else ""
    browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "chrome").strip() or "chrome"
    if cookie_file:
        if not os.path.isfile(cookie_file):
            raise FileNotFoundError(
                "YTDLP_COOKIES_FILE is set, but that file does not exist. "
                "Set it to a Netscape-format cookies.txt file or clear the setting."
            )
        ydl_opts["cookiefile"] = cookie_file
    else:
        ydl_opts["cookiesfrombrowser"] = (browser,)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True) #Extract and return the information dictionary for the URL
            filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
    except yt_dlp.utils.DownloadError as error:
        error_text = str(error)
        if "Could not copy Chrome cookie database" in error_text:
            raise RuntimeError(
                f"yt-dlp cannot read the {browser.title()} cookie database while that "
                f"browser is open. Close {browser.title()} completely and retry, or set "
                "YTDLP_COOKIES_FILE to a local Netscape-format cookies.txt file."
            ) from error
        if "Failed to decrypt with DPAPI" in error_text:
            raise RuntimeError(
                "yt-dlp can read the browser cookie database but Windows would not "
                "decrypt its cookies. Use a Netscape-format cookies.txt export and set "
                "YTDLP_COOKIES_FILE in .env to that file."
            ) from error
        raise

    return filename   

# Dual Audio -> Mono Audio + Any Hz -> 16 KHz
def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path) #this AudioSegment automatically decide which type of file we are taking(mp3, mp4, m4a)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz > here we are setting it to mono audio and 16KhZ

    audio.export(output_path, format="wav")
    return output_path

#CHUNKING
def chunk_audio(wav_path : str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path) #loading 
     #chunking works in miliseconds > so we have to *60*1000
    chunk_ms = chunk_minutes * 60 * 1000


    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format = 'wav') #to save

        chunks.append(chunk_path)

    return chunks

#print(chunk_audio(data_final))


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks


# ['downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_0.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_1.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_2.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_3.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_4.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_5.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project inPython_converted.wav_chunk_6.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_7.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_8.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_9.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_10.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_11.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_12.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_13.wav', 
#  'downloads\\AI Video AssistantWith RAG ｜ Full Project in Python_converted.wav_chunk_14.wav', 
#  'downloads\\AI Video Assistant With RAG｜ Full Project in Python_converted.wav_chunk_15.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_16.wav', 
#  'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_17.wav',
# 'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_18.wav', 
# 'downloads\\AI Video Assistant With RAG ｜ Full Project in Python_converted.wav_chunk_19.wav']


#now we can send this full into Wisper AI to transcribe our audio to text

