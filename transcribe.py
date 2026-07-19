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
# Each audio frame is 30ms long so 30ms * 33 =  990ms roughly a second of silence
SILENCE_LENGTH = 33
# Minimum number of frames containing speech.
SPEECH_MIN_LENGTH = 20

class Transcribe:
    def __init__(self, model_name="base.en", debug_on=False):
        # Controls whether or not we print out debugging messages
        self.__debug_on = debug_on
        # Controls whether audio should be captured from the stream
        self.__capture_audio = True
        # Reference to audio stream & producer
        self.__audio_stream = None
        self.__audio_producer = None
        # Reference to audio data collected from the stream
        self.__audio_data = b""
        # State variables used to process audio stream
        self.__has_spoken = False
        self.__continued_silence_count = 0
        self.__speech_frames_count = 0
        # Create instance of Voice Activated Detection (VAD) utility
        self.__vad = webrtcvad.Vad()
        self.__vad.set_mode(3)
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

        # Create pyAudio instance and initialize audio stream. 
        self.__audio_producer = pyaudio.PyAudio()
        self.__audio_stream = self.__audio_producer.open(
            format=pyaudio.paInt16,
            channels=NB_CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        # Run audio stream producer in a separate thread
        self.__audio_capture_thread = threading.Thread(target=self.__producer_thread)
        self.__audio_capture_thread.start()
        # Prepare the transcriber to run on a separate thread
        self.__transcriber_thread = threading.Thread(target=self.__consumer_thread)

    # Destructor
    def __del__(self):
        self.__capture_audio = False
        # Bring threads to foreground
        self.__audio_capture_thread.join()
        self.__transcriber_thread.join()
        # Release I/O resource
        self.__audio_stream.close()
        self.__audio_producer.terminate()
    
    # Start capturing Audio from microphone
    def proceed(self):
        self.__capture_audio = True
        self.__audio_stream.start_stream()
        # re-create the audio capture thread if it has completed
        self.__audio_capture_thread = threading.Thread(target=self.__producer_thread)
        self.__audio_capture_thread.start()
        self.__print('Creating new producer thread')

    # Pauses audio capture
    def pause(self):
        self.__capture_audio = False
        # Bring thread to foreground for completion.
        self.__audio_capture_thread.join()
        self.__audio_stream.stop_stream()

    # Returns the queue with transcribed audio
    def get_transcriptions(self):
        return self.__transcribed_audio
    
    # Processes audio stream by detecting any speech, sanatizing any extra audio silence
    # And determining when to save the audio stream data into a clip to be transcribed.
    def __process_audio_sream(self):

        # Determnies whether to save audio data into transcriber queue
        should_save_audio_data = False
        # Read 30ms of raw audio data
        chunk = self.__audio_stream.read(CHUNK)
        self.__audio_data += chunk
        # VAD only works with 30ms audio frames
        is_speech: bool = self.__vad.is_speech(chunk, RATE)

        if is_speech:
            if self.__has_spoken == False:
                self.__print('Listening...')
                self.__has_spoken = True
                # Removes any previous silence
                self.__audio_data = chunk
            self.__speech_frames_count += 1
            # Reset silence counter since speech was detected
            self.__continued_silence_count = 0
        else:
            # Increment on silence
            self.__continued_silence_count += 1

        # Close off audio clip after silence is detected
        if (self.__has_spoken and self.__continued_silence_count >= SILENCE_LENGTH):
            # Only save audio clips with meaningful speech 
            if (self.__speech_frames_count >= SPEECH_MIN_LENGTH):
                should_save_audio_data = True
            # Reset counters
            self.__continued_silence_count = 0
            self.__speech_frames_count = 0
            # Reset speech detection flag
            self.__has_spoken = False

        return should_save_audio_data
    
    # Transcribes audio clips from queue into text
    def __consumer_thread(self):
        # Transcribe all the pending audio clips
        while not self.__pending_audio.empty():
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

        self.__print("Consumer thread ended.")
    
    # Processes the audio stream and creates audio clips to be transcribed later
    def __producer_thread(self):
        
        self.__print("Microphone initialized, recording started...")

        while self.__capture_audio:

            if self.__process_audio_sream():
                self.__print('Saving into audio queue...')
                self.__pending_audio.put(self.__audio_data)
                # Run fresh transcriber thread
                if (self.__transcriber_thread.is_alive() and self.__transcriber_thread.native_id == None):
                    self.__transcriber_thread.start()
                else:
                    # Create a new one if last one has finished
                    self.__transcriber_thread = threading.Thread(target=self.__consumer_thread)
                    self.__transcriber_thread.start()
            
        self.__print("producer thread ended.")
    
    # Prints out a debugging message
    def __print(self, message):
        if self.__debug_on:
            print(message)