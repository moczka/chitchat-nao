'''
    Name:   Javier Steven Guerrero
    Date:   August 9th, 2026

    Makes the robot respond to touch

'''
from ..language_model import send_message

class ReactToTouch():
    def __init__(self, session):

        # Get ALMemory and ALTextToSpeech 
        self.memory_service = session.service("ALMemory")
        self.animated_speech = session.service("ALAnimatedSpeech")
        # Listen to touch events from robot sensors
        self.touch = self.memory_service.subscriber("TouchChanged")
        self.id = self.touch.signal.connect(self.__onTouched, "TouchChanged")

    def __onTouched(self, strVarName, value):
        # Disconnect to the event when talking,
        # to avoid repetitions
        self.touch.signal.disconnect(self.id)

        touched_bodies = []
        for p in value:
            if p[1]:
                touched_bodies.append(p[0])

        self.say(touched_bodies)

        # Reconnect again to the event
        self.id = self.touch.signal.connect(self.__onTouched, "TouchChanged")

    def __say(self, bodies):
        if (bodies == []):
            return

        sentence = "My " + bodies[0]

        for b in bodies[1:]:
            sentence = sentence + " and my " + b

        if (len(bodies) > 1):
            sentence = sentence + " are"
        else:
            sentence = sentence + " is"
        sentence = sentence + " touched."

        self.tts.say(sentence)