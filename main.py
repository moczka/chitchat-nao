'''
    Name:   Arjun Pramanik
    Date:   June 9, 2026

'''
from transcribe import Transcribe
from language_model import send_message
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
anim_text_to_speech = session.service("ALAnimatedSpeech")
anim_text_to_speech.setMode("contextual")
# Create memory instance (used to subscribe to events)
memory = session.service("ALMemory")
# Create a notification manager instance
notif_manager = session.service("ALNotificationManager")

# TO-DO: Create a module for the Naoqi Framework integration.
# TO-DO: Create a module that reacts to touch events and generates a prompt
# to solicit a response from the robot. Format in "I am [action] your [part]"

def main():
    global transcriber
    # Subscribe to notifications to mute diagnostic reports
    memory.subscriber("ALNotificationManager/NotificationAdded").signal.connect(on_notification_added)
    try:
         # Set up transcribing tool
        transcriber = Transcribe(on_transcription_complete=process_user_prompt)
        # Prompt user
        print('Listening... Ask Pazuzu anything.')
    except KeyboardInterrupt:
        print('Exiting...')
        exit(1)

def on_notification_added(notif_data):
    # Mute hardware diagnotistic notifications by removing them.
    notif_manager.removeNotification(notif_data["id"])

def process_user_prompt(prompt):
    print(f"User: {prompt}")
    print('\nThinking...\n')
    # Stop capturing audio while SLM generates an answer
    transcriber.pause()
    robot_resp = send_message(prompt)
    print(f"\nRobot: {robot_resp}\n")
    anim_text_to_speech.say(robot_resp)
    # Re-enable transcriber
    transcriber.proceed()

if __name__ == "__main__":
    main()
