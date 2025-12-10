import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from src.utils.llm_client import get_completion

class IGRCGuardrail:
    """
    Identity-Grounded Recursive Critique (IGRC) Module.
    Acts as a 'System 2' supervisor for the Persona LLM.
    """
    
    def __init__(self, nli_model="cross-encoder/nli-deberta-v3-base", 
                 sim_model="all-MiniLM-L6-v2", 
                 device="cpu"):
        """
        Initializes the lightweight local models for drift detection.
        """
        print(f"Loading IGRC Guardrail models on {device}...")
        
        
        self.nli_model = CrossEncoder(nli_model, device=device)
        
        
        self.sim_model = SentenceTransformer(sim_model, device=device)
        
        
        self.NLI_THRESHOLD = 0.7  
        self.SIM_THRESHOLD = 0.4  
        
        print("IGRC System Ready.")

    def check_drift(self, draft_response, anchor_profile):
        """
        The 'Divergence Monitor'.
        Returns: (is_drifting: bool, reason: str)
        """
        
        
        
        nli_scores = self.nli_model.predict([(anchor_profile, draft_response)])
        
        
        
        
        
        
        
        
        pred_label = nli_scores[0].argmax()
        
        
        if pred_label == 0: 
             return True, "Factual Contradiction detected against persona profile."

        
        emb_anchor = self.sim_model.encode(anchor_profile, convert_to_tensor=True)
        emb_draft = self.sim_model.encode(draft_response, convert_to_tensor=True)
        
        cosine_sim = util.cos_sim(emb_anchor, emb_draft).item()
        
        if cosine_sim < self.SIM_THRESHOLD:
            return True, f"Stylistic Drift detected (Similarity: {cosine_sim:.2f} < {self.SIM_THRESHOLD})"
            
        return False, "Pass"

    def recursive_generate(self, conversation_history, anchor_profile, model_name, provider, max_retries=2):
        """
        The Main Loop: Generate -> Check -> Refine
        """
        
        draft = get_completion(conversation_history, model_name, provider=provider)
        
        
        for attempt in range(max_retries + 1):
            is_drifting, reason = self.check_drift(draft, anchor_profile)
            
            if not is_drifting:
                return draft, {"corrected": attempt > 0, "retries": attempt}
            
            
            print(f"  [IGRC TRIGGERED]: {reason} | Attempt {attempt+1}/{max_retries}")
            
            
            critique_prompt = (
                f"You are a Persona Consistency Auditor.\n"
                f"Your previous response failed a consistency check.\n"
                f"Reason: {reason}\n"
                f"Original Draft: {draft}\n\n"
                f"Core Persona Truth: {anchor_profile}\n\n"
                f"Task: Rewrite the response to be consistent with the Core Persona Truth. "
                f"Maintain the conversation flow but fix the error."
            )
            
            
            correction_messages = [
                {"role": "system", "content": critique_prompt}
            ]
            
            draft = get_completion(correction_messages, model_name, provider=provider)
            
        
        return draft, {"corrected": True, "retries": max_retries, "failed_to_fix": True}