import pvporcupine
import sounddevice as sd
import numpy as np
import time
import whisper
import pvcobra
import warnings
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.preprocessing import LabelEncoder
import pickle
import soundfile as sf
import serial
import keyboard
import time

serial_port = 'COM5'
baud_rate = 9600
ser = serial.Serial(serial_port, baud_rate)

time.sleep(2)
print("Ready!!")

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

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

model = whisper.load_model("base.en")

def VAD(result, ctime, cnt):

    maxdata.append(result)
    ind = int(2 * cnt)

    if ctime > 1.5 and len(maxdata) > ind:

        data = maxdata[-ind:]
        avg = sum(data) / len(data)

        return avg <= 0.5

    return False

def audio_callback(indata, frames, time_info, status):
    global buffer, recording, recording_buffer, start_time, cnt

    if status:
        print(status, flush=True)

    buffer = np.concatenate((buffer, indata.flatten()))

    while len(buffer) >= porcupine.frame_length:

        pcm = buffer[:porcupine.frame_length]
        buffer = buffer[porcupine.frame_length:]
        result = porcupine.process(pcm)

        if result >= 0:

            print("Wake word detected!")

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

def transcribe_audio(data):

    data = data.astype(np.float32) / 32768.0
    
    temp_filename = "temp_audio.wav"
    sf.write(temp_filename, data, samplerate=porcupine.sample_rate)

    result = model.transcribe(temp_filename)
    print("Transcription:", result["text"])
    predict_command(result['text'])

def predict_command(text):
    d = {"for":""}
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=maxlen)
    pred = text_command_model.predict(padded_seq)
    command = label_encoder.inverse_transform([pred.argmax()])
    comm = command[0]
    print(comm)

    if comm == "Forward" or 'forward' in text.lower():
        ser.write(b'1')
        time.sleep(1)
        ser.write(b'0')

    elif comm== "Backward" or 'backward' in text.lower():
        ser.write(b'2')
        time.sleep(1)
        ser.write(b'0')

    elif comm== "Left" or 'left' in text.lower():
        ser.write(b'3')
        time.sleep(1.5)
        ser.write(b'0')

    elif comm== "Right" or 'right' in text.lower():
        ser.write(b'4')
        time.sleep(1.5)
        ser.write(b'0')

    elif comm== "Stop" or 'stop' in text.lower():
        ser.write(b'0')
        time.sleep(1)

    time.sleep(0.2)

def start_listening():

    global porcupine
    porcupine = pvporcupine.create(
        access_key="gXC9vgNKtPuibX5gmxg5YT5/xMhQ6r2OGONQ6bWTMK9X3fVAvMgMJw==",
        keyword_paths=[r"WakeWordModel.ppn"]
    )

    global cobra
    cobra = pvcobra.create(access_key="gXC9vgNKtPuibX5gmxg5YT5/xMhQ6r2OGONQ6bWTMK9X3fVAvMgMJw==")
    
    with sd.InputStream(
        channels=1,
        samplerate=porcupine.sample_rate,
        dtype=np.int16,
        callback=audio_callback
    ):
        print("Listening for wake word...")
        sd.sleep(-1)

if __name__ == "__main__":
    start_listening()