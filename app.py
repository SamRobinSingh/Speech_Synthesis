import pvporcupine
import sounddevice as sd
import numpy as np
import time
import whisper
import pvcobra
import warnings
from tensorflow.keras.models import load_model # type: ignore
import requests
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.preprocessing import LabelEncoder
import pickle
import soundfile as sf
from kokoro import KPipeline
import keyboard
from rich.console import Console
from rich.progress import track
from rich.status import Status
from rich.logging import RichHandler
import httpx
from rich.panel import Panel
import pyttsx3
import logging
import os
import tensorflow as tf

# Set TensorFlow log level (Suppress warnings)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Suppress SciKit-Learn Version Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure Rich Console
console = Console()
logging.basicConfig(level="WARNING", handlers=[RichHandler()])
log = logging.getLogger("rich")

console.print("[bold green]System initializing...[/bold green]")
console.print("[bold cyan]Ready!![/bold cyan]")

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load ML Models
console.print("[yellow]Loading models...[/yellow]")
text_command_model = load_model("text_command_model.h5")

with open("tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

maxlen = 100
buffer = np.array([], dtype=np.int16)
recording = False
recording_buffer = []
start_time = None
maxdata = []
cnt = 0

# Load Whisper Model
model = whisper.load_model("small.en")
console.print("[bold cyan]Models loaded successfully![/bold cyan]")

OLLAMA_URL = "http://localhost:11434/api/generate"  # Default Ollama API endpoint
MODEL_NAME = "llama3.2:1b"  # Change this based on your model

def transcribe_audio(data):
    data = data.astype(np.float32) / 32768.0
    
    temp_filename = "temp_audio.wav"
    sf.write(temp_filename, data, samplerate=porcupine.sample_rate)

    # Show processing message
    with console.status("[bold cyan]Transcribing audio...[/]"):
        result = model.transcribe(temp_filename)
    
    transcription = result["text"]
    console.print(Panel(f"[bold yellow]Transcription:[/] {transcription}", title="📢 Speech-to-Text", expand=False))

    # Send transcription to Ollama and get response
    response_text = send_to_ollama(transcription)
    
    console.print(Panel(f"\n[bold green]Ollama Response:[/] {response_text}", title="🤖 AI Assistant", expand=False))

    text_to_speech(response_text)

def send_to_ollama(text):
    prompt = "You are a simple personal assistant. Keep responses short and precise. Ensure the reply is concise and quick. The user prompt is "
    payload = {"model": MODEL_NAME, "prompt": prompt + text, "stream": False}
    
    with httpx.Client(timeout=5) as client:  # Set timeout
        response = client.post(OLLAMA_URL, json=payload)
    
    return response.json().get("response", "No response from Ollama.")


def text_to_speech(text, engine_type="offline"):
    """Convert text to speech and play it."""
    
    pipeline = KPipeline(lang_code='a',device='cuda')

    generator = pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+')

    for _, _, audio in generator:
        audio_np = np.array(audio, dtype=np.float32)
        sd.play(audio_np, samplerate=24000)
        sd.wait()

def VAD(result, ctime, cnt):
    maxdata.append(result)
    ind = int(2 * cnt)
    if ctime > 1.5 and len(maxdata) > ind:
        data = maxdata[-ind:]
        avg = sum(data) / len(data)
        return avg <= 0.1
    return False

def audio_callback(indata, frames, time_info, status):
    global buffer, recording, recording_buffer, start_time, cnt

    if status:
        #log.warning(status)
        pass

    buffer = np.concatenate((buffer, indata.flatten()))

    while len(buffer) >= porcupine.frame_length:
        pcm = buffer[:porcupine.frame_length]
        buffer = buffer[porcupine.frame_length:]
        result = porcupine.process(pcm)

        if result >= 0:
            console.print("[bold magenta]Wake word detected![/bold magenta]")
            recording = True
            recording_buffer = []
            maxdata = []
            cnt = 0
            start_time = time.time()

        if recording:
            result = cobra.process(pcm)
            curr_time = time.time() - start_time

            if curr_time < 0.5:
                cnt += 1
                continue

            recording_buffer.append(indata.copy())

            if VAD(result, curr_time, cnt):
                recording = False
                audio_data = np.concatenate(recording_buffer)
                transcribe_audio(audio_data)

def start_listening():

    global porcupine
    porcupine = pvporcupine.create(
        access_key="gXC9vgNKtPuibX5gmxg5YT5/xMhQ6r2OGONQ6bWTMK9X3fVAvMgMJw==",
        keyword_paths=[r"WakeWordModel1.ppn"]
    )

    global cobra
    cobra = pvcobra.create(access_key="gXC9vgNKtPuibX5gmxg5YT5/xMhQ6r2OGONQ6bWTMK9X3fVAvMgMJw==")
    
    with sd.InputStream(
        channels=1,
        samplerate=porcupine.sample_rate,
        dtype=np.int16,
        callback=audio_callback
    ):
        console.print("[bold green]Listening for wake word...[/bold green]")
        with Status("[cyan]Waiting for wake word...[/cyan]"):
            sd.sleep(-1)

if __name__ == "__main__":
    start_listening()
