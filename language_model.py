'''
Name:   Javier S. Guerrero
Date:   04/19/2026

Sets up the small language model

'''
from llama_cpp import Llama

MODEL_PATH="./models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"

SETUP_CONFIG='''You are a robot assistant for the Computer Club at Quincy College. 
Your name is Pazuzu. Do not say you are a text-based AI, say you are a robot.
You have a head. You have arms. You have legs. Therefore, you have a physical body that experiences the real world.
You provide short answers.''' # noqa

# Initialize the model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,  # Context window size
    n_threads=10, # Adjust based on your CPU cores
    n_batch=1024, # Increase available memory 
    verbose=False
)

# Create a chat history
messages = [
    {"role": "system", "content": SETUP_CONFIG} 
]

def send_message(user_input):
    try: 
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        
        # Extract model reply
        reply = response['choices'][0]['message']['content']
        
        # Add reply to history for context
        messages.append({"role": "assistant", "content": reply})
        
        return reply
    
    except:
        print("Language model errored out...")
        return ""

# Run if called as script
if __name__ == "__main__":
    while True:
        prompt = input("Type a message: ")
        print(f"\n{send_message(prompt)}\n")
