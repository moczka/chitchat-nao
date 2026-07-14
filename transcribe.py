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
# Each audio frame is 30ms long so 30ms * 50 = roughly a second and a half of silence
SILENCE_LENGTH = 50
# Minimum number of frames containing speech.
SPEECH_MIN_LENGTH = 20

# TODO: Only start a consumer thread when there are items in the pending_audio queue
# TODO: Implement PyAudio non-blocking mode to remove the need of the producer_thread https://people.csail.mit.edu/hubert/pyaudio/docs/#id4
# TODO: Make use of the stream.strart_strean() and stream.stop_stream() methods instead of __should_listen https://people.csail.mit.edu/hubert/pyaudio/docs/


class Transcribe:
    def __init__(self, model_name="base.en", debug_on=False):
        # Controls whether or not we print out debugging messages
        self.__debug_on = debug_on
        # Reference to audio stream & producer
        self.__audio_stream = None
        self.__audio_producer = None
        # Controls whether audio stream is opened or not.
        self.__should_listen = False
        # Stores the audio clips to be transcribed
        self.__pending_audio = queue.Queue()
        # Stores transcribed audio clips
        self.__transcribed_audio = queue.Queue()
        # Download model
        self.__print('Downloading Whisper model...')
        try:
            self.__model = WhisperModel(model_name, device="cpu", compute_type="int8")
            self.__print('Whisper model downloaded successfully!')
        except:
            self.__print('Failed to download Whisper model.')

    # Destructor
    def __del__(self):
        # Release I/O resource
        if (self.__audio_stream != None and self.__audio_producer):
            self.__audio_stream.close()
            self.__audio_producer.terminate();

    # Initializes capturing audio and transcription process (should only be called once)
    def init(self):
        self.__should_listen = True
        producer = threading.Thread(target=self.__producer_thread)
        producer.start()
        consumer = threading.Thread(target=self.__consumer_thread)
        consumer.start()
    
    # Start capturing Audio from microphone
    def start(self):
        self.__should_listen = True
    # Pauses audio capture
    def stop(self):
        self.__should_listen = False

    # Returns the queue with transcribed audio
    def get_transcriptions(self):
        return self.__transcribed_audio
    
    # Transcribes audio clips from queue into text
    def __consumer_thread(self):
        while True:
            if not self.__pending_audio.empty():
                self.__print('Transcribing audio clip...')
                # Convert raw audio byte data into numpy array for model
                audio_clip: np.ndarray = np.frombuffer(self.__pending_audio.get(), np.int16).astype(np.float32) / 255.0
                # Transcribe text
                segments, info = self.__model.transcribe(audio_clip, language="en", beam_size=5)
                # https://medium.com/@venn5708/two-important-libraries-used-for-audio-processing-and-streaming-in-python-d3b718a75904
                user_message = ""
                for segment in segments:
                    self.__print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                    user_message += segment.text
                # Place transcribed audio into queue
                if (user_message != ""):
                    self.__transcribed_audio.put(user_message)
    
    # Processes the audio stream and creates audio clips to be transcribed later
    def __producer_thread(self):
        # State variables
        audio_data = b""
        has_spoken = False
        continued_silence_count = 0
        speech_frames_count = 0
        # Create instance of Voice Activated Detection
        vad = webrtcvad.Vad()
        vad.set_mode(3)
        self.__audio_producer = pyaudio.PyAudio()
        self.__audio_stream = self.__audio_producer.open(
            format=pyaudio.paInt16,
            channels=NB_CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        self.__print("Microphone initialized, recording started...")

        while True:
            # Read 30ms of raw audio data
            chunk = self.__audio_stream.read(CHUNK)
            audio_data += chunk
            # VAD only works with 30ms audio frames
            is_speech: bool = vad.is_speech(chunk, RATE)

            if is_speech:
                if has_spoken == False:
                    self.__print('Listening...')
                    has_spoken = True
                    # Removes any previous silence
                    audio_data = chunk
                speech_frames_count += 1
                # Reset silence counter
                continued_silence_count = 0
            else:
                # Increment on silence
                continued_silence_count += 1

            # Close off audio clip after silence is detected
            if (has_spoken and continued_silence_count >= SILENCE_LENGTH):
                self.__print("About to save audio clip...")
                # Store audio clip in queue
                if (self.__should_listen and speech_frames_count >= SPEECH_MIN_LENGTH):
                    self.__print('Saving into audio queue...')
                    self.__pending_audio.put(audio_data)
                # Reset counters
                continued_silence_count = 0
                speech_frames_count = 0
                # Reset speech detection
                has_spoken = False
    
    # Prints out a debugging message
    def __print(self, message):
        if self.__debug_on:
            print(message)