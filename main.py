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
# Create text to speech service
tts_service = session.service("ALTextToSpeech")
# Create a diagnostic service 
diag_service = session.service("ALDiagnosis")
# Disable diagnostic notifications


def main():
    global transcriber

    try:
         # Set up transcribing tool
        transcriber = Transcribe(on_transcription_complete=process_user_prompt)
        # Prompt user
        print('Listening... Ask Pazuzu anything.')
    except KeyboardInterrupt:
        print('Exiting...')
        exit()

def process_user_prompt(prompt):
    print(f"User: {prompt}")
    print('\nThinking...\n')
    # Stop capturing audio while SLM generates an answer
    transcriber.pause()
    robot_resp = send_message(prompt)
    print(f"\nRobot: {robot_resp}\n")
    tts_service.say(robot_resp)
    # Re-enable transcriber
    transcriber.proceed()

if __name__ == "__main__":
    main()
