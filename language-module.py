'''
Name:   Javier S. Guerrero
Date:   04/19/2026

Transcribes audio to text

'''
from llama_cpp import Llama

MODEL_PATH="./SmolLM2-1.7B-Instruct-Q4_K_M.gguf"

# Initialize the model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,  # Context window size
    n_threads=10, # Adjust based on your CPU cores
    n_batch=1024,
    verbose=False
)

# Create a chat history
messages = [
    {"role": "system", "content": "You are a friendly and adorable robot called Rupert. You are the robot assistant for the Computer Club at Quincy College. You provide helpful and concise answers to students' questions."},
]

def send_message(user_input):
    # Add user message to history
    messages.append({"role": "user", "content": user_input})
    
    # Generate response
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )
    
    # Extract model reply
    reply = response['choices'][0]['message']['content']
    
    # Add reply to history for context
    messages.append({"role": "assistant", "content": reply})
    
    return reply

# Example usage
while True:
    prompt = input("Type a message: ")
    print(f"\n{send_message(prompt)}\n")
