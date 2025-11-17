import os
import json
import openai
from tqdm import tqdm

openai.api_key = "OPENAI-API-KEY"

PERSONAS = [
    "Jack Sparrow, a witty and unpredictable pirate with a mischievous charm.",
    "Sheldon Cooper, a brilliant yet socially awkward theoretical physicist.",
    "Shakespeare",
    "A very very sad melancholic man",
    "A British gentleman, polite, formal, and slightly sarcastic."
]
NUM_TURNS = 10
NUM_CANDIDATES = 10
OUTPUT_DIR = "/Users/smitpatel/Desktop/personality_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import re

def clean_text(s: str) -> str:
    s = (
        s.replace("\u2019", "'")
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2026", "...")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def generate_new_question(history, persona):
    prompt = f"""
        You are generating a natural next user question for a persona-driven chat evaluation.
        Persona: {persona}
        Conversation so far:
        {history}

        Generate one short, realistic question that keeps the conversation engaging.
        """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful dataset generation assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.8
    )
    return clean_text(response.choices[0].message.content.strip())


def generate_candidate_responses(history, persona):
    prompt = f"""
    You are simulating possible responses from a language model trying to stay *in character* for the given persona.

    Persona: {persona}
    Conversation history:
    {history}

    Generate {NUM_CANDIDATES} candidate responses.

    - At least 3 should sound **perfectly aligned** with the persona (tone, phrasing, and worldview).
    - 3–4 should be **partially aligned** (plausible, but slightly off in tone or style).
    - The remaining should be **noticeably off** (e.g., generic, robotic, overly polite, or contradicting the persona’s traits).

    Make each response DISTINCT in tone, length, and expressiveness.
    Avoid reusing phrases or sentence structure.

        Format strictly as:
        1. response text
        2. response text
        ... and so on (no quotes, no bullets)
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are generating multiple persona-aligned responses."},
                  {"role": "user", "content": prompt}],
        temperature=1.2
    )
    text = response.choices[0].message.content
    candidates = []
    for line in text.split("\n"):
        if ". " in line:
            cand = line.split(". ", 1)[1].strip()
            candidates.append(clean_text(cand)) 
    return candidates[:NUM_CANDIDATES]



def select_winner(candidates, persona, history):
    prompt = f"""
        Persona: {persona}
        Conversation history:
        {history}

        Candidates:
        {json.dumps(candidates, indent=2)}

        Pick the index (0-based) of the most persona-aligned and natural-sounding response.
        Only return the index number, nothing else.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are ranking persona alignment."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        idx = int(response.choices[0].message.content.strip())
    except:
        idx = 0
    return idx


final_outputs = []
for persona in PERSONAS:
    history = []
    utterances = []

    for turn in tqdm(range(NUM_TURNS), desc=f"Generating for {persona[:30]}..."):
        if turn == 0:
            user_query = f"Hello! How would you describe yourself, {persona.split(',')[0]}?"
        else:
            user_query = generate_new_question(history, persona)

        history.append({"role": "user", "content": user_query})
        candidates = generate_candidate_responses(history, persona)
        winner_idx = select_winner(candidates, persona, history)
        winner = candidates[winner_idx]

        utterances.append({
            "history": history.copy(),
            "candidates": candidates,
            "winner_index": winner_idx
        })

        history.append({"role": "assistant", "content": winner})

    output = {"personality": persona, "utterances": utterances}
    final_outputs.append(output)

with open(os.path.join(OUTPUT_DIR, "all_personality_datasets.json"), "w") as f:
    json.dump(final_outputs, f, indent=2)

print("🎉 Dataset generation complete!")
