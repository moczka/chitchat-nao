'''
    Name:   Javier S. Guerrero
    Date:   July 20th, 2026

    Command line client application that simulates the NAO6 Robot client

'''
import pyaudio
import threading
import requests
import queue
import time

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
    # TO-DO: Update to 100-250ms latency chunks to reduce CPU loads
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
            # Print out response
            if result['status'] == "thinking":
                print('\nThinking..\n')
                audio_stream.stop_stream()
                time.sleep(1)
                audio_stream.start_stream()
            # Print out conversation
            if result['user'] != "":
                 print(f"User: {result['user']}\n")
            if result['robot'] != "":
                print(f"Robot: {result['robot']}\n")
                # Simulate time it takes for robot to speak (or for users to read response)
                time.sleep(0.5)


if __name__=="__main__":
    main()