'''
Name:   Javier S. Guerrero
Date:   04/19/2026

Transcribes audio to text

'''
import whisper

model = whisper.load_model('tiny.en')


result = model.transcribe('test_snip.wav', language="en", fp16=False)

print(result["text"]);

