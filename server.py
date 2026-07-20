'''
    Name:   Javier Steven Guerrero
    Date:   July 19th, 2026

    Server application to process audio stream from NAO6 Robot
'''
from faster_whisper import WhisperModel
import asyncio
import uvicorn
from fastapi import FastAPI, Request

HOST = "localhost"
PORT = 3000
app = FastAPI()
MODEL_NAME = "base.en"

@app.post("/listen")
async def process_audio_stream(request: Request):
    audio_data: bytes = await request.body()

    return {
        "text": "hello"
    }


if __name__=="__main__":

    # Download Whisper model
    model: WhisperModel = None
    print('Downloading Whisper model...')
    try:
        model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
        print('Whisper model downloaded successfully!')
    except:
        print('Failed to download Whisper model.')

    uvicorn.run(app, host=HOST, port=PORT)