from __future__ import print_function
import qi

ROBOT_IP = "192.168.50.33"
PORT = 9559

app = qi.Application([
    "say_nao_py2.py",
    "--qi-url=tcp://{}:{}".format(ROBOT_IP, PORT)
])

app.start()
session = app.session

tts = session.service("ALTextToSpeech")
tts.say("Hello. This is the laptop controlling NAO.")

print("Laptop-to-NAO TTS worked.")
