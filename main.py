'''
    Name:   Arjun Pramanik
    Date:   June 9, 2026

'''
import functools

from transcribe import Transcribe
from language_model.model import generate_response
from nao6_modules.react_to_touch import ReactToTouch

import qi
import sys

transcriber = None
PORT = "9561"
ROBOT_IP = "192.168.50.33"

app = qi.Application(
    sys.argv,
    url=f"tcp://{ROBOT_IP}:{PORT}"
)
app.start()
session = app.session
# Create text to speech instance
animated_speech = session.service("ALAnimatedSpeech")
# Create memory instance (used to subscribe to events)
memory = session.service("ALMemory")
# Create a notification manager instance
notif_manager = session.service("ALNotificationManager")
# Notification publisher
notif_pub = memory.subscriber("notificationAdded")

# TO-DO: Create a module for the Naoqi Framework integration.
# TO-DO: Create a module that reacts to touch events and generates a prompt
# to solicit a response from the robot. Format in "I am [action] your [part]"

def main():
    global transcriber, session, notif_pub
    # Subscribe to notifications to mute diagnostic reports
    notif_pub.signal.connect(functools.partial(on_notification_added, "notificationAdded"))
    # Instantiate custom modules
    react_to_touch = ReactToTouch(session)
    try:
         # Set up transcribing tool
        transcriber = Transcribe(on_transcription_complete=process_user_prompt)
        # Prompt user
        print('Listening... Ask Pazuzu anything.')
    except KeyboardInterrupt:
        print('Exiting...')
        exit(1)

def on_notification_added(event_name, notif_id):
    # Mute hardware diagnotistic notifications by removing them.
    notif_manager.remove(notif_id)
    

def process_user_prompt(prompt):
    print(f"User: {prompt}")
    print('\nThinking...\n')
    # Stop capturing audio while SLM generates an answer
    transcriber.pause()
    robot_resp = generate_response(prompt)
    print(f"\nRobot: {robot_resp}\n")
    animated_speech.say(robot_resp)
    # Re-enable transcriber
    transcriber.proceed()

if __name__ == "__main__":
    main()
