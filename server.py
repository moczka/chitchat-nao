'''
    Name:   Javier Steven Guerrero
    Date:   July 19th, 2026

    Server application to process audio stream from NAO6 Robot
'''
from transcribe import Transcribe
from language_model import send_message
import queue
import uvicorn
from fastapi import FastAPI, Request

HOST = "localhost"
PORT = 3000

robot_responses = queue.Queue()

status = "listening"
app = FastAPI()

@app.post("/listen")
async def process_audio_stream(request: Request):

    audio_data: bytes = await request.body()
    transcriber.transcribe(audio_data)
    response = {'status': status, 'user': '', 'robot': ''}

    if not user_prompts.empty():
        response['user'] = user_prompts.get()
    if not robot_responses.empty():
        robot_resp = robot_responses.get()
        response['robot'] = robot_resp
    
    return response

def process_user_prompts(user_message):
    global status
    status = "thinking"
    robot_resp = send_message(user_message)
    robot_responses.put(robot_resp)
    status = "listening"

transcriber = Transcribe(server_mode=True, callback=process_user_prompts)
# Reference to transcribed user prompts from audio
user_prompts = transcriber.get_transcriptions()

if __name__=="__main__":
    uvicorn.run(app, host=HOST, port=PORT)