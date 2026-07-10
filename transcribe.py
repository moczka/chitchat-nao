'''
Name:   Javier S. Guerrero
Date:   04/19/2026

Transcribes audio to text
'''
from faster_whisper import WhisperModel
import webrtcvad
import pyaudio
import wave
import queue
import threading

NB_CHANNELS = 1
RATE = 16000
CHUNK = 480 # To generate 30ms frames
# Each audio frame is 30ms long so 30ms * 33 = roughly one second.
SILENCE_LENGTH = 33

audio_queue = queue.Queue()

def producer_thread():
    global audio_queue
    # State variables
    recording_count = 0
    audio_data = b""
    has_spoken = False
    silence_count = 0
    # Create instance of Voice Activated Detection
    vad = webrtcvad.Vad()
    vad.set_mode(3)
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=NB_CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,    # 1 second of audio
    )

    print("-" * 80)
    print("Microphone initialized, recording started...")
    print("-" * 80)
    print("TRANSCRIPTION")
    print("-" * 80)

    while True: 
        
        chunk = stream.read(CHUNK) # Read 1 second of audio data
        audio_data += chunk
        is_speech: bool = vad.is_speech(chunk, RATE)
        # Increment on silence
        if is_speech:
            has_spoken = True
            silence_count = 0
        else:
            silence_count += 1

        # Close off recording after a 1-second silence.
        if (has_spoken and silence_count >= SILENCE_LENGTH):
            audio_queue.put(audio_data)
            # Reset count
            silence_count = 0
            # Reset speech detection
            has_spoken = False
            # Increment recording count
            recording_count = recording_count + 1
            # Store audio clip in queue
            audio_queue.put(audio_data)
            # Write recording to file
            print("Writing to disk...")
            with wave.open(f"recordings/audio_{recording_count}.wav", 'wb') as wav_file:
                wav_file.setnchannels(NB_CHANNELS)
                wav_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wav_file.setframerate(RATE)
                wav_file.writeframes(audio_data)

            # Reset binary string
            audio_data = b""

if __name__ == "__main__":

    model_size = "base.en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # print("Starting transcription...")

    # segments, info = model.transcribe('processed_audio_edilson.wav', language="en", beam_size=5)

    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    # # https://medium.com/@venn5708/two-important-libraries-used-for-audio-processing-and-streaming-in-python-d3b718a75904
    # for segment in segments:
    #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    producer = threading.Thread(target=producer_thread)
    producer.start()