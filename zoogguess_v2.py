import os
import sys
import gc
import json
import signal
import atexit
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================================
# 1. CLEANUP HANDLERS
# =====================================================================
model = None
tokenizer = None

def cleanup():
    global model, tokenizer
    if model is not None or tokenizer is not None:
        print("\n[ZoogGuess] Unloading model from RAM/VRAM...")
        del model
        del tokenizer
        model = None
        tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[ZoogGuess] Memory flushed cleanly.")

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

atexit.register(cleanup)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =====================================================================
# 2. NEURAL QUESTION GENERATOR (SmolLM2-135M)
# =====================================================================
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading local neural model ({MODEL_ID}) onto {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None
)
if DEVICE == "cpu":
    model.to("cpu")
model.eval()

# Natural human phrasing translations
NATURAL_TEMPLATES = {
    "class": "Is this animal a {val}?",
    "biome": "Does this animal typically live in {val}?",
    "diet": "Is this creature known for being a {val}?",
    "lifestyle": "Is this animal known for {val}?",
    "covering": "Does this creature have {val}?",
    "limbs": "Does this animal have {val}?",
    "weapons": "Does this animal have {val}?",
    "head": "Does this creature feature {val}?",
    "build": "Does this animal have a {val}?",
    "special": "Is this animal known for {val}?"
}

FRIENDLY_PHRASES = {
    "four legs": "walking on four legs",
    "cloven hooves": "having split or cloven hooves",
    "single solid hooves": "having solid hooves like a horse",
    "domesticated": "being a domesticated animal or livestock",
    "social or herd-dwelling": "living in herds, packs, or social groups",
    "solitary": "living mostly alone rather than in a herd",
    "herbivore or plant-eater": "eating mostly grass, leaves, or plants",
    "carnivore or active predator": "hunting and eating other animals",
    "pastures or farmland": "farms, barns, or agricultural pastures",
    "temperate forests or woodlands": "woodlands or temperate forests",
    "polar ice or cold tundra": "freezing arctic ice or cold tundra",
    "tropical rainforest or jungle": "tropical rainforests or jungles",
    "arid desert or scrubland": "dry deserts or arid scrublands",
    "mountainous or alpine highlands": "steep mountains or rocky highlands",
    "thick wool or fleece": "a thick coat of wool or fleece",
    "sharp fangs or canine teeth": "prominent fangs or sharp canine teeth",
    "tufted ears": "distinctive tufts of hair on the tips of its ears",
    "horns or blunt defense": "horns on its head"
}

def generate_natural_question(category, trait_val):
    clean_val = trait_val.replace("_", " ").strip()
    friendly_val = FRIENDLY_PHRASES.get(clean_val, clean_val)
    
    template = NATURAL_TEMPLATES.get(category, "Does your animal feature {val}?")
    direct_q = template.format(val=friendly_val)
    
    # Prompt SmolLM2 to polish into fluent conversational English
    prompt = (
        "Rephrase this into a friendly, clear Yes/No question for 20 Questions without changing its meaning. "
        "Keep it strictly under 15 words and end with a question mark:\n\n"
        f"Input: {direct_q}\n"
        "Question:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.2,
            top_p=0.85,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    first_line = generated.split("\n")[0].split("?")[0].strip()
    
    if len(first_line) > 5 and any(first_line.lower().startswith(w) for w in ["is", "does", "can", "would", "has", "do", "are"]):
        return first_line + "?"
    return direct_q

# =====================================================================
# 3. KNOWLEDGE BASE & BAYESIAN ENGINE
# =====================================================================
DB_PATH = "animals_v2_clean.json"
with open(DB_PATH, "r", encoding="utf-8") as f:
    ANIMAL_DB = json.load(f)

ALL_ANIMALS = list(ANIMAL_DB.keys())

def get_animal_traits(animal_data):
    traits = set()
    for cat, val in animal_data.get("attributes", {}).items():
        if isinstance(val, list):
            for v in val:
                if v.lower().strip() not in ["none", "unknown", "n/a", ""]:
                    traits.add((cat, v))
        elif isinstance(val, str):
            if val.lower().strip() not in ["none", "unknown", "n/a", ""]:
                traits.add((cat, val))
    return traits

def calculate_best_trait(probabilities, asked_traits, recent_categories):
    trait_weights = defaultdict(float)
    for animal, prob in probabilities.items():
        if prob < 0.0002:
            continue
        for t in get_animal_traits(ANIMAL_DB[animal]):
            if t not in asked_traits:
                trait_weights[t] += prob

    best_trait = None
    best_score = float("inf")

    for trait, weight in trait_weights.items():
        # Trait must cut between 10% and 90% of current probability mass
        if not (0.10 <= weight <= 0.90):
            continue

        cat, _ = trait
        diff = abs(weight - 0.5)

        # STRICT CATEGORY LOCKOUT: heavily penalize recently asked categories
        if cat in recent_categories:
            penalty = 0.25 if (recent_categories and cat == recent_categories[-1]) else 0.12
            diff += penalty

        if diff < best_score:
            best_score = diff
            best_trait = trait

    # If all balanced traits are locked, fallback to any available valid trait
    if not best_trait and trait_weights:
        best_trait = min(trait_weights.keys(), key=lambda t: abs(trait_weights[t] - 0.5))

    return best_trait

def run_game():
    n = len(ALL_ANIMALS)
    probs = {animal: 1.0 / n for animal in ALL_ANIMALS}
    asked_traits = set()
    recent_categories = []
    question_count = 0

    print("\n" + "=" * 65)
    print("   ZOOGGUESS V2: REFINED 500-ANIMAL ENGINE")
    print(f" Loaded {n} animals | SLM Generation: Active ({DEVICE})")
    print(" Answer options:")
    print("  [y] Yes      [p] Probably / Mostly")
    print("  [n] No       [d] Doubtful / Unlikely    [k] Don't Know")
    print("=" * 65 + "\n")

    CONFIDENCE_MAP = {
        "y": 0.88, "yes": 0.88,
        "p": 0.72, "probably": 0.72,
        "k": 0.50, "dont know": 0.50, "idk": 0.50,
        "d": 0.28, "doubtful": 0.28,
        "n": 0.12, "no": 0.12
    }

    while question_count < 20:
        sorted_candidates = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_animal, top_prob = sorted_candidates[0]
        second_animal, second_prob = sorted_candidates[1]

        # Stop threshold
        if top_prob > 0.80 or (top_prob > 0.60 and top_prob > second_prob * 3.5):
            break

        best_trait = calculate_best_trait(probs, asked_traits, recent_categories[-2:])
        if not best_trait:
            break

        asked_traits.add(best_trait)
        cat, val = best_trait
        recent_categories.append(cat)
        question_count += 1

        q_text = generate_natural_question(cat, val)

        while True:
            raw_resp = input(f"[Q{question_count}] {q_text}\n  (y/p/k/d/n): ").strip().lower()
            if raw_resp in CONFIDENCE_MAP:
                p_yes = CONFIDENCE_MAP[raw_resp]
                break
            print("Invalid input. Choose: y, p, k, d, or n.")

        # Bayesian probability update
        for animal in ALL_ANIMALS:
            has_trait = best_trait in get_animal_traits(ANIMAL_DB[animal])
            likelihood = p_yes if has_trait else (1.0 - p_yes)
            probs[animal] *= likelihood

        # Normalize mass
        total_mass = sum(probs.values())
        if total_mass > 0:
            probs = {k: v / total_mass for k, v in probs.items()}

        top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        preview = ", ".join([f"{ANIMAL_DB[k]['display_name']} ({v*100:.1f}%)" for k, v in top3 if v > 0.03])
        print(f"       Leaders: {preview}")
        print("-" * 55)

    sorted_candidates = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    winner, confidence = sorted_candidates[0]

    print("\n" + "=" * 65)
    print(f" DEDUCTION COMPLETE in {question_count} questions!")
    print(f" Guess: **{ANIMAL_DB[winner]['display_name'].upper()}** (Confidence: {confidence*100:.1f}%)")
    print(f" Summary: {ANIMAL_DB[winner]['description'][:280]}...")
    print("=" * 65)

if __name__ == "__main__":
    run_game()
