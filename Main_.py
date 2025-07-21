from kokoro import KPipeline
import sounddevice as sd
import numpy as np

pipeline = KPipeline(lang_code='a', model=False)  

while True:
    text = input("Enter text (or type 'exit' to quit): ")
    if text.lower() == "exit":
        break  

    generator = pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+')

    for _, _, audio in generator:
        audio_np = np.array(audio, dtype=np.float32)
        sd.play(audio_np, samplerate=24000)
        sd.wait() 
