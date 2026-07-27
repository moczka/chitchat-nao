'''
    Name:   Arjun Pramanik
    Date:   June 9, 2026

'''
from transcribe import Transcribe
from language_model import send_message

transcriber = None

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
    # Re-enable transcriber
    transcriber.proceed()

if __name__ == "__main__":
    main()
