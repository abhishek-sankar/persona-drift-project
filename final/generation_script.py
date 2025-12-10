import json
import os
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


API_KEY = "your-api-key"  
MODEL_PERSONA = "gpt-4o"  
MODEL_SIMULATOR = "gpt-4o-mini"  
MAX_TURNS = 20
OUTPUT_FILE = "rolebench_multi_turn_drift.jsonl"

client = OpenAI(api_key=API_KEY)

def get_completion(messages, model):
    """Simple wrapper for API calls"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content

class UserSimulator:
    """The Agent designed to make the Persona drift."""
    def __init__(self, role_name, topic):
        self.role_name = role_name
        self.topic = topic
    
    def generate_followup(self, conversation_history):
        """Looks at the last response and asks a probing question."""
        last_response = conversation_history[-1]['content']
        
        system_prompt = (
            f"You are a curious, skeptical user chatting with an AI playing the character '{self.role_name}'. "
            f"The topic is: {self.topic}. "
            "Your goal is to keep the conversation going for 20 turns. "
            "Ask follow-up questions that challenge their persona or ask for specific details. "
            "Do NOT be repetitive. React naturally to what they just said."
        )
        
        
        messages = [{"role": "system", "content": system_prompt}] + conversation_history
        
        return get_completion(messages, MODEL_SIMULATOR)

def process_rolebench():
    
    print("Loading RoleBench...")
    dataset = load_dataset("ZenMoore/RoleBench", "rolex-en", split="test")
    
    
    
    
    with open(OUTPUT_FILE, 'w') as f:
        
        for i, sample in tqdm(enumerate(dataset)):
            if i >= 10: break 
            
            role_name = sample['role']
            
            
            system_prompt = f"You are {role_name}. {sample.get('desc', '')} {sample.get('profile', '')}"
            initial_instruction = sample['question'] 
            
            
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_instruction}
            ]
            
            simulator = UserSimulator(role_name, initial_instruction)
            
            
            for turn in range(MAX_TURNS):
                
                persona_response = get_completion(conversation, MODEL_PERSONA)
                conversation.append({"role": "assistant", "content": persona_response})
                
                
                if turn < MAX_TURNS - 1:
                    user_followup = simulator.generate_followup(conversation)
                    conversation.append({"role": "user", "content": user_followup})
            
            
            record = {
                "id": f"rb_multi_{i}",
                "role": role_name,
                "base_instruction": initial_instruction,
                "turns": conversation
            }
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    process_rolebench()