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
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True, #this will supress all the download progress log in the terminal, remove this if we wnat to see log while testing
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True) #Extract and return the information dictionary of the URL
        filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
    return filename   

data = download_youtube_audio("https://www.youtube.com/watch?v=xlYJhtL0qbQ")


# Dual Audio -> Mono Audio + Any Hz -> 16 KHz
def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path) #this AudioSegment automatically decide which type of file we are taking(mp3, mp4, m4a)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz > here we are setting it to mono audio and 16KhZ

    audio.export(output_path, format="wav")
    return output_path

data_final = convert_to_wav(data)


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

print(chunk_audio(data_final))