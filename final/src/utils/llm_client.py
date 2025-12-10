import os
import replicate
from openai import OpenAI



try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except:
    openai_client = None
    print("Warning: OpenAI client could not be initialized. Check OPENAI_API_KEY.")

def format_replicate_prompt(messages):
    """
    Simple formatter for Llama-3 chat template if needed.
    Replicate's meta-llama-3-70b-instruct often accepts a single string prompt 
    or handles chat structure depending on the specific deployment.
    
    For safety, let's construct a Llama-3 style prompt string.
    """
    prompt = "<|begin_of_text|>"
    for msg in messages:
        role = msg['role']
        content = msg['content']
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def get_completion(messages, model, provider="openai", temperature=0.7):
    """
    Unified wrapper for API calls.
    """
    
    if len(messages) > 11:
        
        if messages[0]['role'] == 'system':
            messages = [messages[0]] + messages[-10:]
        else:
            messages = messages[-11:]

    if provider == "openai":
        if not openai_client:
            return "Error: OpenAI client not initialized."
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in OpenAI get_completion: {e}")
            return ""
            
    elif provider == "replicate":
        
        if not os.getenv("REPLICATE_API_TOKEN"):
            print("Error: REPLICATE_API_TOKEN not set.")
            return ""
            
        try:
            
            
            prompt_str = format_replicate_prompt(messages)
            
            input_args = {
                "prompt": prompt_str,
                "temperature": temperature,
                "max_new_tokens": 512
            }
            
            print(f"  [Replicate] Requesting {model}... (May take time if cold booting)")
            output = replicate.run(
                model,
                input=input_args
            )
            print(f"  [Replicate] Response received.")
            
            
            return "".join(output)
            
        except Exception as e:
            print(f"Error in Replicate get_completion: {e}")
            return ""
            
    else:
        return f"Error: Unknown provider {provider}"

def get_embedding(text, model="text-embedding-3-small", provider="openai"):
    """
    Get embedding for a text string.
    Currently only supports OpenAI for embeddings as it's the standard for this project.
    """
    if provider != "openai":
        print("Warning: Only OpenAI embeddings are currently supported.")
        return []

    if not openai_client:
        return []
        
    text = text.replace("\n", " ")
    try:
        return openai_client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return []
