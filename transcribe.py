'''
Name:   Javier S. Guerrero
Date:   04/19/2026

Transcribes audio to text
'''
from faster_whisper import WhisperModel
import webrtcvad
import pyaudio
import numpy as np
import queue
import threading

NB_CHANNELS = 1 # Mono audio (single channel)
RATE = 16000
CHUNK = 480 # To generate 30ms audio frames
# Each audio frame is 30ms long so 30ms * 33 = roughly one second of silence
SILENCE_LENGTH = 33

audio_queue = queue.Queue()

def consumer_thread(model):

    while True:

        if not audio_queue.empty():
            # Convert raw audio byte data into numpy array for model
            audio_clip: np.ndarray = np.frombuffer(audio_queue.get(), np.int16).astype(np.float32) / 255.0
            # Transcribe text
            print('Transcribing audio clip...')
            segments, info = model.transcribe(audio_clip, language="en", beam_size=5)
            # https://medium.com/@venn5708/two-important-libraries-used-for-audio-processing-and-streaming-in-python-d3b718a75904
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


def producer_thread():
    global audio_queue
    # State variables
    audio_data = b""
    has_spoken = False
    continued_silence_count = 0
    # Create instance of Voice Activated Detection
    vad = webrtcvad.Vad()
    vad.set_mode(3)
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=NB_CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    print("Microphone initialized, recording started...")

    while True: 
        # Read 30ms of raw audio data
        chunk = stream.read(CHUNK)
        audio_data += chunk
        # VAD only works with 30ms audio frames
        is_speech: bool = vad.is_speech(chunk, RATE)

        if is_speech:
            if has_spoken == False:
                has_spoken = True
                # Removes any previous silence
                audio_data = chunk
            # Reset silence counter
            continued_silence_count = 0
        else:
            # Increment on silence
            continued_silence_count += 1

        # Close off recording after a 1-second silence.
        if (has_spoken and continued_silence_count >= SILENCE_LENGTH):
            # Reset count
            continued_silence_count = 0
            # Reset speech detection
            has_spoken = False
            # Store audio clip in queue
            print('Saving into audio queue...')
            audio_queue.put(audio_data)

if __name__ == "__main__":

    print('Downloading Whisper model...')

    model_size = "base.en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    producer = threading.Thread(target=producer_thread)
    producer.start()

    consumer = threading.Thread(target=consumer_thread, args=[model])
    consumer.start()