'''
    Name:   Javier Steven Guerrero
    Date:   August 9th, 2026

    Makes the robot respond to touch

'''
import functools
import re
from language_model.model import generate_response

PART_LABEL_TO_NAME = {
    "Head": "head",
    "LArm": "left arm",
    "RArm": "right arm",
    "LHand": "left hand",
    "RHand": "right hand",
    "LFoot": "left foot",
    "RFoot": "right foot",
    "ChestBoard": "chest"
}

class ReactToTouch():
    def __init__(self, session):
        # Get ALMemory and ALTextToSpeech proxies
        self.memory_service = session.service("ALMemory")
        self.animated_speech = session.service("ALAnimatedSpeech")
        # Listen to touch events from robot sensors
        self.touch = self.memory_service.subscriber("TouchChanged")
        self.id = self.touch.signal.connect(functools.partial(self.__on_touched, "TouchChanged"))

    def __on_touched(self, event_name, body_parts):
        # Disconnect to the event when talking, to avoid repetitions
        print("On touch was called.")
        self.touch.signal.disconnect(self.id)
        # TODO: Use http://doc.aldebaran.com/2-8/family/nao_technical/contact-sensors_naov6.html#naov6-contact-hand
        # To compose a string that can be used to generate a response from the language model.
        print(body_parts)
        for body_part in body_parts:
            if body_part[1]:
                # Solicit a response from the robot
                self.animated_speech.say(self.__generate_response(body_part[0]))
                break
        #Reconnect to handle other touch events
        self.id = self.touch.signal.connect(functools.partial(self.__on_touched, "TouchChanged"))

    def __generate_response(body_part_info):
        info = body_part_info.split("/")
        # Gather the information regarding the body part
        part_label = info[0]
        part_location = info[2] if len(info) > 2 else ""
        # Compose prompt for language model
        verb = "grabbing" if re.search('Arm', part_label) else "rubbing"
        prompt = ""
        if part_location != "":
            prompt = f"I am {verb} your {PART_LABEL_TO_NAME[part_label]} from the {part_location.lower()} gently."
        else:
            prompt = f"I am {verb} your {PART_LABEL_TO_NAME[part_label]} gently."
        # Generate response
        return generate_response(prompt)

    def pause_reactions(self):
        '''Pauses responding to touch interactions.'''
        self.touch.signal.disconnect(self.id)

    def continue_reactions(self):
        '''Continues responding to physical touch interactions.'''
        self.id = self.touch.signal.connect(self.__on_touched, "TouchedChanged")