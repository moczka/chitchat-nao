"""
    Author:     Javier Steven Guerrero
    Date:       08/04/2026

    Example of using ALTextToSpeech module
"""
import qi
import sys


app = qi.Application(
    sys.argv,
    url="tcp://192.168.50.33:9561"
)
app.start()
session = app.session

tts_service = session.service("ALTextToSpeech")
tts_service.say("Hello there!")

tts_service.say("Here is another message")
