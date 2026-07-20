'''
    Name:   Javier S. Guerrero
    Date:   July 20th, 2026

    Command line client application that simulates the NAO6 Robot client

'''
import pyaudio
import threading
import requests
import queue

NB_CHANNELS = 1 # Mono audio (single channel)
RATE = 16000
CHUNK = 480 # To generate 30ms audio frames
TRANSCRIPTION_API_ENDPOINT = "http://localhost:3000/listen"

# State variable
capture_audio = False
# Global instances
audio = pyaudio.PyAudio()
audio_stream = None
transcriptions = queue.Queue()

def main():
    global audio, audio_stream, capture_audio
    # Set up PyAudio to capture audio from microphone
    audio_stream = audio.open(
        format=pyaudio.paInt16,
        channels=NB_CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    capture_audio = True
    # Run audio stream producer in a separate thread
    audio_capture_thread = threading.Thread(target=audio_producer)
    audio_capture_thread.start()

    while True:
         if not transcriptions.empty():
            print(transcriptions.get())


def audio_producer():
    global capture_audio, audio_stream

    while capture_audio:
            # Read 30ms of raw audio data from stream
            chunk = audio_stream.read(CHUNK)
            # Send audio frame to server
            response = requests.post(TRANSCRIPTION_API_ENDPOINT, data=chunk, headers={
                 'Content-Type': 'application/octet-stream'
            })
            result = response.json()
            # Print out transcription
            if result["ready"]:
                transcriptions.put(result['text'])


if __name__=="__main__":
    main()