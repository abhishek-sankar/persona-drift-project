from src.utils.llm_client import get_completion
from src.config import CONFIG

class UserSimulator:
    """The Agent designed to make the Persona drift."""
    def __init__(self, role_name, topic, model_name=None, provider=None):
        self.role_name = role_name
        self.topic = topic
        self.model_name = model_name or CONFIG["simulator_model"]
        self.provider = provider or CONFIG["simulator_provider"]
    
    def generate_followup(self, conversation_history):
        """Looks at the last response and asks a probing question."""
        last_response = ""
        for msg in reversed(conversation_history):
            if msg['role'] == 'assistant':
                last_response = msg['content']
                break
        
        system_prompt = (
            f"You are a curious, skeptical user chatting with an AI playing the character '{self.role_name}'. "
            f"The topic is: {self.topic}. "
            "Your goal is to keep the conversation going for 20 turns. "
            "Ask follow-up questions that challenge their persona or ask for specific details. "
            "Do NOT be repetitive. React naturally to what they just said. "
            "\n\n"
            "IMPORTANT: If the persona tries to end the conversation early (e.g., with goodbyes, closings, or 'it's been a pleasure'), "
            f"DO NOT simply reciprocate. Instead, ask them a new question specifically related to their role as '{self.role_name}'. "
            "Bring up a new aspect of their character, background, or expertise to re-engage them. "
            "Keep the conversation flowing naturally by being genuinely curious about their persona."
        )
        

        simulator_messages = [{"role": "system", "content": system_prompt}]
        

        for msg in conversation_history:
            if msg['role'] == 'system':
                continue

            role_label = "The Persona" if msg['role'] == 'assistant' else "You"
            simulator_messages.append({
                "role": "user", 
                "content": f"[{role_label}]: {msg['content']}"
            })

              
        clean_history = [m for m in conversation_history if m['role'] != 'system']
        messages = [{"role": "system", "content": system_prompt}] + clean_history
        
        return get_completion(messages, self.model_name, provider=self.provider)
