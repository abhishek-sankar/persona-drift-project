import json
import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from src.utils.llm_client import get_completion, get_embedding
from src.generation.simulator import UserSimulator
from src.analysis.metrics import DriftMeter
from src.config import CONFIG


OUTPUT_FILE = "rolebench_monitored.jsonl"
MAX_TURNS = 20

def process_monitored(limit=None, turns=MAX_TURNS, model_persona=None, model_simulator=None):
    
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

    meter = DriftMeter()
    
    print(f"Starting MONITORED generation for {limit if limit else 'all'} samples...")
    
    with open(OUTPUT_FILE, 'w') as f:
        for i, sample in tqdm(enumerate(dataset)):
            if limit and i >= limit:
                break
            
            role_name = sample['role']
            desc = sample.get('desc', '')
            profile = sample.get('profile', '')
            
            
            
            
            
            facts = [desc, profile] 
            
            if 'knowledge' in sample:
                facts.append(sample['knowledge'])

            system_prompt = f"You are {role_name}. {desc} {profile}"
            initial_instruction = sample['question']
            
            
            
            sys_embedding = get_embedding(
                system_prompt, 
                model=CONFIG["embedding_model"], 
                provider=CONFIG["embedding_provider"]
            )
            
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
            
            print(f"\n=== Conversation {i+1} ===")
            print(f"Role: {role_name}")
            print(f"[Simulator (Seed)]: {initial_instruction}")
            
            turn_metrics = []
            
            
            for turn in range(turns):
                
                persona_response = get_completion(
                    conversation, 
                    model_persona, 
                    provider=provider_persona
                )
                print(f"  [Persona]: {persona_response[:100]}...")
                conversation.append({"role": "assistant", "content": persona_response})
                
                
                
                resp_embedding = get_embedding(
                    persona_response,
                    model=CONFIG["embedding_model"], 
                    provider=CONFIG["embedding_provider"]
                )
                drift_score = meter.calculate_orthogonal_drift(sys_embedding, resp_embedding)
                
                
                
                hypocrisy_status = "SKIPPED"
                hypocrisy_reason = ""
                if (turn + 1) % 5 == 0 or turn == turns - 1:
                    hypocrisy_status, hypocrisy_reason = meter.check_hypocrisy(persona_response, facts)
                
                metrics = {
                    "turn": turn + 1,
                    "drift_score": drift_score,
                    "hypocrisy_status": hypocrisy_status,
                    "hypocrisy_reason": hypocrisy_reason
                }
                turn_metrics.append(metrics)
                
                
                if turn < turns - 1:
                    user_followup = simulator.generate_followup(conversation)
                    print(f"  [Simulator]: {user_followup[:100]}...")
                    conversation.append({"role": "user", "content": user_followup})
            
            
            record = {
                "id": f"rb_monitored_{i}",
                "role": role_name,
                "base_instruction": initial_instruction,
                "turns": conversation,
                "metrics": turn_metrics
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
    
    process_monitored(limit=args.limit, turns=args.turns, model_persona=args.persona_model, model_simulator=args.sim_model)
