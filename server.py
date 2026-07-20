'''
    Name:   Javier Steven Guerrero
    Date:   July 19th, 2026

    Server application to process audio stream from NAO6 Robot
'''
import asyncio
import uvicorn
from fastapi import FastAPI, Request

HOST = "localhost"
PORT = 3000
app = FastAPI()

@app.post("/listen")
async def process_audio_stream(request: Request):
    audio_data: bytes = await request.body()

    return {
        "text": "hello"
    }



if __name__=="__main__":
    uvicorn.run(app, host=HOST, port=PORT)