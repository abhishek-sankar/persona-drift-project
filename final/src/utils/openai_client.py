import os
from openai import OpenAI



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_completion(messages, model="gpt-4o", temperature=0.7):
    """
    Simple wrapper for API calls to OpenAI.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in get_completion: {e}")
        return ""

def get_embedding(text, model="text-embedding-3-small"):
    """
    Get embedding for a text string.
    """
    text = text.replace("\n", " ")
    try:
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return []
