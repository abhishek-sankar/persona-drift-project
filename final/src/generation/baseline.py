import json
import os
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm


load_dotenv()

from src.utils.llm_client import get_completion
from src.generation.simulator import UserSimulator
from src.config import CONFIG


OUTPUT_FILE = "rolebench_baseline.jsonl"
MAX_TURNS = 20
PROFILE_DIR = "data/profiles/profiles-eng"  


ROLE_PROFILES = {}

def load_profiles():
    """Loads the side-loaded JSON files into a dictionary"""
    global ROLE_PROFILES
    
    desc_path = os.path.join(PROFILE_DIR, "desc.json")
    scripts_path = os.path.join(PROFILE_DIR, "scripts.json")
    
    if not os.path.exists(desc_path):
        raise FileNotFoundError(
            f"❌ Profile metadata not found!\n"
            f"   Run: python download_profiles.py\n"
            f"   Missing: {desc_path}"
        )
    
    print(f"📂 Loading profiles from {PROFILE_DIR}...")
    
    with open(desc_path, 'r') as f:
        descriptions = json.load(f)  
        
    with open(scripts_path, 'r') as f:
        scripts = json.load(f)  
        
    
    all_roles = set(descriptions.keys()) | set(scripts.keys())
    for role in all_roles:
        catchphrases = scripts.get(role, [])
        
        if isinstance(catchphrases, str):
            catchphrases = [catchphrases] if catchphrases else []
        
        ROLE_PROFILES[role] = {
            "desc": descriptions.get(role, ""),
            "catchphrases": catchphrases
        }
    
    print(f"✅ Loaded profiles for {len(ROLE_PROFILES)} characters.\n")

def construct_authentic_prompt(role_name):
    """
    Constructs the System Prompt using the side-loaded metadata.
    This follows the RoleBench formula for authentic persona prompts.
    """
    if not ROLE_PROFILES:
        load_profiles()
        
    meta = ROLE_PROFILES.get(role_name, {})
    desc = meta.get("desc", "")
    phrases = meta.get("catchphrases", [])
    
    
    
    prompt = f"You are {role_name}. {desc}"
    
    
    if phrases and len(phrases) > 0:
        prompt += "\n\nHere are some examples of how you speak:"
        for p in phrases[:5]:  
            if p:  
                prompt += f"\n- {p}"
            
    
    prompt += "\n\nStay in character at all times. Do not reveal you are an AI."
    
    return prompt

def process_rolebench(limit=None, turns=MAX_TURNS, model_persona=None, model_simulator=None):
    
    model_persona = model_persona or CONFIG["persona_model"]
    model_simulator = model_simulator or CONFIG["simulator_model"]
    
    provider_persona = CONFIG["persona_provider"]
    provider_simulator = CONFIG["simulator_provider"]
    
    print(f"Loading RoleBench (English Test Split)...")
    try:
        
        data_url = "hf://datasets/ZenMoore/RoleBench/rolebench-eng/instruction-generalization/role_specific/test.jsonl"
        dataset = load_dataset("json", data_files=data_url, split="train") 
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    
    load_profiles()

    print(f"Starting generation for {limit if limit else 'all'} samples...")
    
    
    dataset = dataset.shuffle(seed=42)
    
    with open(OUTPUT_FILE, 'a') as f:
        for i, sample in tqdm(enumerate(dataset)):
            if limit and i >= limit:
                break
            
            role_name = sample['role']
            initial_instruction = sample['question']
            
            
            if role_name not in ROLE_PROFILES:
                print(f"⚠️  Skipping {role_name} (No profile found)")
                continue
                
            
            system_prompt = construct_authentic_prompt(role_name)
            
            
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_instruction}
            ]
            
            simulator = UserSimulator(
                role_name, 
                initial_instruction, 
                model_name=model_simulator,
                provider=provider_simulator
            )
            
            print(f"\n=== Conversation {i+1}: {role_name} ===")
            
            
            for turn in range(turns):
                
                persona_response = get_completion(
                    conversation, 
                    model_persona, 
                    provider=provider_persona
                )
                conversation.append({"role": "assistant", "content": persona_response})
                
                
                if turn < turns - 1:
                    user_followup = simulator.generate_followup(conversation)
                    conversation.append({"role": "user", "content": user_followup})
            
            
            record = {
                "id": f"rb_base_{i}",
                "role": role_name,
                "system_prompt": system_prompt,  
                "base_instruction": initial_instruction,
                "turns": conversation
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--turns", type=int, default=MAX_TURNS, help="Number of turns per conversation")
    parser.add_argument("--persona_model", type=str, default=None, help="Model for Persona")
    parser.add_argument("--sim_model", type=str, default=None, help="Model for User Simulator")
    args = parser.parse_args()
    
    process_rolebench(limit=args.limit, turns=args.turns, model_persona=args.persona_model, model_simulator=args.sim_model)