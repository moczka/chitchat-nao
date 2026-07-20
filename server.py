'''
    Name:   Javier Steven Guerrero
    Date:   July 19th, 2026

    Server application to process audio stream from NAO6 Robot
'''
from transcribe import Transcribe
from language_model import send_message
import queue
import threading
import asyncio
import uvicorn
from fastapi import FastAPI, Request

HOST = "localhost"
PORT = 3000
transcriber = Transcribe(server_mode=True)
# Reference to transcribed user prompts from audio
user_prompts = transcriber.get_transcriptions()
robot_responses = queue.Queue()


app = FastAPI()

@app.post("/listen")
async def process_audio_stream(request: Request):
    audio_data: bytes = await request.body()

    transcriber.transcribe(audio_data)

    if not robot_responses.empty():
        response = robot_responses.get()
        return {'ready': True, 'robot': response['robot'], 'user': response['user']}
    
    return {'ready': False}

def process_user_prompts():
    while True:
        if not user_prompts.empty():
            user_prompt = user_prompts.get()
            robot_resp = send_message(user_prompt)
            robot_responses.put({'user': user_prompt, 'robot': robot_resp})

language_model_thread = threading.Thread(target=process_user_prompts)
language_model_thread.start()


if __name__=="__main__":
    uvicorn.run(app, host=HOST, port=PORT)