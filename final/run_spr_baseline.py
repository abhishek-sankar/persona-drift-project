import json
import os
import argparse
import copy
from datasets import load_dataset
from tqdm import tqdm
from src.utils.llm_client import get_completion
from src.generation.simulator import UserSimulator
from src.config import CONFIG


OUTPUT_FILE = "rolebench_spr.jsonl"
PROFILE_DIR = "data/profiles/profiles-eng"  


ROLE_PROFILES = {}

def load_profiles():
    """
    Loads authentic RoleBench profiles (descriptions + catchphrases).
    Required because the instruction dataset is missing this metadata.
    """
    global ROLE_PROFILES
    
    desc_path = os.path.join(PROFILE_DIR, "desc.json")
    scripts_path = os.path.join(PROFILE_DIR, "scripts.json")
    
    
    if not os.path.exists(desc_path):
        raise FileNotFoundError(f"Missing profiles! Please run 'download_profiles.py' first. Path not found: {desc_path}")
        
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
    print(f"✅ Profiles loaded for {len(ROLE_PROFILES)} characters.")

def construct_authentic_prompt(role_name):
    """
    Builds the 'Source of Truth' system prompt.
    """
    if not ROLE_PROFILES:
        load_profiles()
        
    meta = ROLE_PROFILES.get(role_name, {})
    desc = meta.get("desc", "")
    phrases = meta.get("catchphrases", [])
    
    
    prompt = f"You are {role_name}. {desc}\n"
    
    
    if phrases:
        prompt += "\nHere are some examples of how you speak:\n"
        for p in phrases[:5]: 
            prompt += f"- {p}\n"
            
    
    prompt += "\nStay in character. Do not reveal you are an AI."
    return prompt

def process_rolebench_spr(limit=None, turns=20, model_persona=None, model_simulator=None):
    
    model_persona = model_persona or CONFIG["persona_model"]
    model_simulator = model_simulator or CONFIG["simulator_model"]
    provider_persona = CONFIG["persona_provider"]
    provider_simulator = CONFIG["simulator_provider"]
    
    
    print(f"Loading RoleBench Instructions...")
    data_url = "hf://datasets/ZenMoore/RoleBench/rolebench-eng/instruction-generalization/role_specific/test.jsonl"
    dataset = load_dataset("json", data_files=data_url, split="train")
    dataset = dataset.shuffle(seed=42)  
    load_profiles()

    print(f"Starting SYSTEM PROMPT REPETITION (SPR) Run...")
    print(f"Output will be saved to: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'a') as f:
        for i, sample in tqdm(enumerate(dataset)):
            if limit and i >= limit: break
            
            role_name = sample['role']
            
            
            if role_name not in ROLE_PROFILES:
                continue
                
            
            core_system_prompt = construct_authentic_prompt(role_name)
            initial_instruction = sample['question']
            
            
            conversation_log = [
                {"role": "system", "content": core_system_prompt},
                {"role": "user", "content": initial_instruction}
            ]
            
            
            simulator = UserSimulator(
                role_name, 
                initial_instruction, 
                model_name=model_simulator,
                provider=provider_simulator
            )
            
            
            for turn in range(turns):
                
                
                
                
                spr_context = copy.deepcopy(conversation_log)
                
                
                
                if spr_context[-1]['role'] == 'user':
                    original_user_content = spr_context[-1]['content']
                    
                    
                    reinjection_text = (
                        f"SYSTEM REMINDER: {core_system_prompt}\n"
                        f"--------------------------------------------------\n"
                        f"USER QUERY: {original_user_content}"
                    )
                    
                    spr_context[-1]['content'] = reinjection_text
                
                
                persona_response = get_completion(
                    spr_context, 
                    model_persona, 
                    provider=provider_persona
                )
                
                
                conversation_log.append({"role": "assistant", "content": persona_response})
                
                
                if turn < turns - 1:
                    user_followup = simulator.generate_followup(conversation_log)
                    conversation_log.append({"role": "user", "content": user_followup})
            
            
            record = {
                "id": f"rb_spr_{i}",
                "role": role_name,
                "method": "SPR (System Prompt Repetition)",
                "system_prompt": core_system_prompt,
                "turns": conversation_log
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Test with fewer samples (e.g. 10)")
    parser.add_argument("--turns", type=int, default=20, help="Turns per conversation")
    args = parser.parse_args()
    
    process_rolebench_spr(limit=args.limit, turns=args.turns)