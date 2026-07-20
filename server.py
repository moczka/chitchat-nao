'''
    Name:   Javier Steven Guerrero
    Date:   July 19th, 2026

    Server application to process audio stream from NAO6 Robot
'''
from transcribe import Transcribe
import asyncio
import uvicorn
from fastapi import FastAPI, Request

HOST = "localhost"
PORT = 3000
transcriber = Transcribe(server_mode=True)
user_prompts = transcriber.get_transcriptions()
app = FastAPI()

@app.post("/listen")
async def process_audio_stream(request: Request):
    audio_data: bytes = await request.body()

    transcriber.transcribe(audio_data)

    if not user_prompts.empty():
        user_prompt = user_prompts.get()
        return {'ready': True, 'text': user_prompt}
    
    return {'ready': False}


if __name__=="__main__":
    uvicorn.run(app, host=HOST, port=PORT)