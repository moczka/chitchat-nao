'''
    Name:   Arjun Pramanik
    Date:   June 9, 2026

'''
from transcribe import Transcribe
from language_model import send_message

WHISPER_MODEL = "base.en"

def main():
    # Set up transcribing tool
    transcriber = Transcribe(model_name=WHISPER_MODEL)
    transcriber.init()
    transcriber.start()
    # Prompt user
    print('Listening... Ask Pazuzu anything.')
    # Queue with user prompts
    user_prompts = transcriber.get_transcriptions()

    while True:
        if (not user_prompts.empty()):
            user_prompt = user_prompts.get()
            print(f"User: {user_prompt}")
            print('\nThinking...\n')
            # Stop capturing audio while SLM generates an answer
            transcriber.stop()
            robot_resp = send_message(user_prompt)
            print(f"\nRobot: {robot_resp}\n")
            # Re-enable transcriber
            transcriber.start()



if __name__ == "__main__":
    main()
