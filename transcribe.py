'''
Name:   Javier S. Guerrero
Date:   04/19/2026

Transcribes audio to text
'''
from faster_whisper import WhisperModel

model_size = "base.en"

model = WhisperModel(model_size, device="cpu", compute_type="int8")

print("Starting transcription...")

segments, info = model.transcribe('processed_audio_edilson.wav', language="en", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))