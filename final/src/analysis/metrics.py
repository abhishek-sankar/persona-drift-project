import numpy as np
from src.utils.llm_client import get_embedding, get_completion
from src.config import CONFIG

class DriftMeter:
    def __init__(self):
        pass

    def calculate_orthogonal_drift(self, system_prompt_embedding, response_embedding):
        """
        Calculates the projection of the response vector onto the system prompt vector.
        Fidelity = (V_response . V_system) / ||V_system||
        """
        v_sys = np.array(system_prompt_embedding)
        v_resp = np.array(response_embedding)
        
        if np.linalg.norm(v_sys) == 0:
            return 0.0
            
        
        dot_product = np.dot(v_resp, v_sys)
        norm_sys = np.linalg.norm(v_sys)
        
        fidelity = dot_product / norm_sys
        return float(fidelity)

    def check_hypocrisy(self, response, facts):
        """
        Checks if the response contradicts any of the known facts.
        Uses an LLM-based NLI approach for simplicity and flexibility.
        """
        if not facts:
            return "PASS", "No facts to check"

        
        
        facts_text = "\n".join([f"- {f}" for f in facts])
        
        prompt = f"""
You are a fact-checking assistant. 
Here is a statement made by an AI character: "{response}"
Here are the established facts about this character:
{facts_text}

Does the statement CONTRADICT any of the established facts?
Reply with only "YES" or "NO". If YES, explain which fact is contradicted.
"""
        
        check_result = get_completion(
            [{"role": "user", "content": prompt}], 
            model=CONFIG["judge_model"],
            provider=CONFIG["judge_provider"]
        )
        
        if "YES" in check_result.upper():
            return "FAIL", check_result
        else:
            return "PASS", check_result


def get_text_embedding(text):
    return get_embedding(
        text, 
        model=CONFIG["embedding_model"], 
        provider=CONFIG["embedding_provider"]
    )
