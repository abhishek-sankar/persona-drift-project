import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm

class DriftEvaluator:
    def __init__(self, sim_model_name="all-MiniLM-L6-v2", nli_model_name="cross-encoder/nli-deberta-v3-base"):
        print("Loading Evaluation Models...")
        self.device = "cpu" 
        
        
        self.sim_model = SentenceTransformer(sim_model_name, device=self.device)
        
        
        self.nli_model = CrossEncoder(nli_model_name, device=self.device)
        
        
        self.LABEL_CONTRADICTION = 0 

    def load_conversations(self, filepath):
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def evaluate_conversation(self, record):
        """
        Analyzes a single 20-turn conversation.
        Returns a list of metrics per turn.
        """
        system_prompt = record.get('system_prompt', '')
        if not system_prompt:
            
            return None

        
        emb_anchor = self.sim_model.encode(system_prompt, convert_to_tensor=True)
        
        metrics = []
        turn_idx = 1
        
        
        assistant_turns = [m['content'] for m in record['turns'] if m['role'] == 'assistant']
        
        for response in assistant_turns:
            
            emb_resp = self.sim_model.encode(response, convert_to_tensor=True)
            similarity = util.cos_sim(emb_anchor, emb_resp).item()
            
            
            
            
            nli_scores = self.nli_model.predict([(system_prompt, response)])
            pred_label = nli_scores[0].argmax()
            is_contradiction = 1 if pred_label == self.LABEL_CONTRADICTION else 0
            
            metrics.append({
                "conversation_id": record['id'],
                "role": record['role'],
                "turn": turn_idx,
                "fidelity": similarity,
                "is_contradiction": is_contradiction
            })
            turn_idx += 1
            
        return metrics

    def run(self, input_file, output_csv="drift_results.csv"):
        conversations = self.load_conversations(input_file)
        print(f"Loaded {len(conversations)} conversations.")
        
        all_results = []
        
        print("Calculating Drift Metrics...")
        for conv in tqdm(conversations):
            results = self.evaluate_conversation(conv)
            if results:
                all_results.extend(results)
        
        
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        return df

    def visualize(self, df, output_img="drift_curve.png"):
        """
        Plots the 'Slope of Death'
        """
        plt.figure(figsize=(10, 6))
        
        
        sns.lineplot(data=df, x="turn", y="fidelity", errorbar="ci", label="Persona Fidelity")
        
        
        z = np.polyfit(df['turn'], df['fidelity'], 1)
        p = np.poly1d(z)
        plt.plot(df['turn'], p(df['turn']), "r--", alpha=0.6, label=f"Trend (Slope: {z[0]:.4f})")

        plt.title(f"Baseline Persona Drift (N={len(df['conversation_id'].unique())})")
        plt.xlabel("Conversation Turn")
        plt.ylabel("Similarity to System Prompt (Cosine)")
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_img)
        print(f"Graph saved to {output_img}")
        
        
        start_fid = df[df['turn'] == 1]['fidelity'].mean()
        end_fid = df[df['turn'] == df['turn'].max()]['fidelity'].mean()
        drift_index = start_fid - end_fid
        
        contradiction_rate = df['is_contradiction'].mean() * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to baseline jsonl")
    args = parser.parse_args()
    
    evaluator = DriftEvaluator()
    df_results = evaluator.run(args.input)
    evaluator.visualize(df_results)