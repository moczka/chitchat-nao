'''
    Name:   Javier Steven Guerrero
    Date:   August 9th, 2026

    Makes the robot respond to touch

'''
from language_model.model import generate_response

class ReactToTouch():
    def __init__(self, session):
        # Get ALMemory and ALTextToSpeech proxies
        self.memory_service = session.service("ALMemory")
        self.animated_speech = session.service("ALAnimatedSpeech")
        # Listen to touch events from robot sensors
        self.touch = self.memory_service.subscriber("TouchChanged")
        self.id = self.touch.signal.connect(self.__onTouched, "TouchChanged")

    def __onTouched(self, event_name, value):
        # Disconnect to the event when talking,
        # to avoid repetitions
        self.touch.signal.disconnect(self.id)

        for (body_part, was_touched) in value:
            if was_touched:
                # Solicit a response from the robot
                self.say(generate_response(
                    f"I am rubbing your {body_part} gently."
                ))
                break
        # Reconnect to handle other touch events
        self.id = self.touch.signal.connect(self.__onTouched, "TouchChanged")