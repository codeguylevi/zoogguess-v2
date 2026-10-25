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
# 1. RESOURCE MANAGEMENT & CLEANUP HANDLERS
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
# 2. AUTOREGRESSIVE SLM INITIALIZATION (SmolLM2-135M-Instruct)
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

# Conversational translator to banish awkward biology terms
FRIENDLY_TRAIT_MAP = {
    # Diets
    "folivore browser": "feeding primarily on leaves, twigs, and tree bark",
    "frugivore": "eating mostly fruit and berries",
    "insectivore": "eating mostly insects and invertebrates",
    "herbivore/grazer": "eating grass and vegetation",
    "carnivore/predator": "hunting and eating other animals",
    "omnivore": "eating both plants and meat",
    "piscivore": "eating primarily fish",
    # Limbs
    "hooves": "walking on hooves",
    "four_legs_quadruped": "walking on all four legs",
    "bipedal_two_legs": "standing and walking on two legs",
    "pectoral_flippers": "having flippers for swimming",
    "prehensile_tail": "having a grasping tail that can hold branches",
    # Anatomy
    "sagittal crest": "having a prominent bony crest on the top of its head",
    "tufted tail": "having a tail with a tuft of hair on the end",
    "permanent horns": "possessing permanent horns on its head",
    "branched antlers": "possessing branched antlers that shed",
    "blubber fat layer": "having a thick layer of insulating blubber",
    "smooth rubbery hairless skin": "having smooth, hairless, rubbery skin",
    # Lifestyle & Biome
    "diurnal_day_active": "being active mostly during daylight hours",
    "nocturnal_night_active": "being active mostly at night",
    "solitary": "living a solitary life rather than in a herd or pack",
    "social_or_pack": "living in herds, packs, or social groups",
    "arboreal tree dweller": "spending most of its life up in the trees",
    "semi aquatic": "spending significant time in the water"
}

def determine_stem(category, clean_trait):
    # Apply friendly translation if present
    for k, v in FRIENDLY_TRAIT_MAP.items():
        if k in clean_trait:
            clean_trait = clean_trait.replace(k, v)

    if category == "biome":
        return f"Does your animal make its home in {clean_trait}"
    elif category == "class":
        return f"Is this creature considered a {clean_trait}"
    elif category in ["weapons", "markings", "covering", "head"]:
        if "having" in clean_trait or "possessing" in clean_trait:
            return f"Is this animal characterized by {clean_trait}"
        return f"Does this animal sport noticeable {clean_trait}"
    elif category == "limbs":
        if "walking" in clean_trait or "moving" in clean_trait or "having" in clean_trait:
            return f"Is your creature known for {clean_trait}"
        return f"Does your creature move around using {clean_trait}"
    elif category in ["diet", "lifestyle", "build"]:
        if "eating" in clean_trait or "feeding" in clean_trait or "being" in clean_trait or "living" in clean_trait:
            return f"Is your animal known for {clean_trait}"
        return f"Would you describe this animal as {clean_trait}"
    return f"Does your animal have {clean_trait}"

def generate_neural_question(category, trait_val):
    clean_trait = trait_val.replace("_", " ").replace("/", " or ")
    stem = determine_stem(category, clean_trait)
    
    prompt = (
        "Complete the following yes-no question about an animal attribute. "
        "Keep it strictly as a single yes-no question. Do not list alternatives or multiple options.\n\n"
        f"Incomplete question: {stem}"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            temperature=0.2,
            top_p=0.8,
            repetition_penalty=1.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    first_clause = completion.split("\n")[0].split("?")[0].strip()
    candidate = f"{stem} {first_clause}".strip()
    if not candidate.endswith("?"):
        candidate += "?"
    candidate = " ".join(candidate.split())

    disallowed_patterns = [", or", "what type", "which of", "how many", "rather than", "if so", "folivore"]
    if any(p in candidate.lower() for p in disallowed_patterns) or candidate.count("?") > 1:
        return f"{stem}?"

    return candidate

# =====================================================================
# 3. KNOWLEDGE BASE & TRAIT EXTRACTION
# =====================================================================
DB_PATH = "animals_v2_clean.json"
with open(DB_PATH, "r", encoding="utf-8") as f:
    ANIMAL_DB = json.load(f)

ALL_ANIMALS = list(ANIMAL_DB.keys())

OPPOSITES = {
    ("lifestyle", "solitary"): ("lifestyle", "social_or_pack"),
    ("lifestyle", "social_or_pack"): ("lifestyle", "solitary"),
    ("lifestyle", "nocturnal_night_active"): ("lifestyle", "diurnal_day_active"),
    ("lifestyle", "diurnal_day_active"): ("lifestyle", "nocturnal_night_active"),
    ("diet", "carnivore/predator"): ("diet", "herbivore/grazer"),
    ("diet", "herbivore/grazer"): ("diet", "carnivore/predator")
}

def get_animal_traits(animal_data):
    traits = set()
    attrs = animal_data.get("attributes", {})
    for cat, val in attrs.items():
        if isinstance(val, list):
            for v in val:
                traits.add((cat, v))
        elif isinstance(val, str):
            traits.add((cat, val))
    return traits

# =====================================================================
# 4. BAYESIAN DEDUCTION ENGINE
# =====================================================================
def calculate_best_trait(probabilities, asked_traits, last_category):
    trait_weights = defaultdict(float)
    for animal, prob in probabilities.items():
        if prob < 0.0005:
            continue
        for t in get_animal_traits(ANIMAL_DB[animal]):
            if t not in asked_traits:
                trait_weights[t] += prob

    best_trait = None
    best_score = float("inf")

    for trait, weight in trait_weights.items():
        if not (0.04 < weight < 0.96):
            continue

        cat, _ = trait
        diff = abs(weight - 0.5)

        if cat == last_category and len(probabilities) > 5:
            diff += 0.08

        if diff < best_score:
            best_score = diff
            best_trait = trait

    return best_trait

def run_game():
    n = len(ALL_ANIMALS)
    probs = {animal: 1.0 / n for animal in ALL_ANIMALS}
    asked_traits = set()
    last_category = None
    question_count = 0

    print("\n" + "=" * 65)
    print("   ZOOGGUESS V2: REFINED 500-ANIMAL ENGINE")
    print(f" Loaded {n} animals | SLM Generation: Active ({DEVICE})")
    print(" Answer options:")
    print("  [y] Yes      [p] Probably / Mostly")
    print("  [n] No       [d] Doubtful / Unlikely    [k] Don't Know")
    print("=" * 65 + "\n")

    CONFIDENCE_MAP = {
        "y": 0.90, "yes": 0.90,
        "p": 0.75, "probably": 0.75,
        "k": 0.50, "dont know": 0.50, "idk": 0.50,
        "d": 0.25, "doubtful": 0.25,
        "n": 0.10, "no": 0.10
    }

    while question_count < 22:
        sorted_candidates = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_animal, top_prob = sorted_candidates[0]
        second_animal, second_prob = sorted_candidates[1]

        if top_prob > 0.82 or (top_prob > 0.65 and top_prob > second_prob * 3.5):
            break

        best_trait = calculate_best_trait(probs, asked_traits, last_category)
        if not best_trait:
            break

        asked_traits.add(best_trait)
        if best_trait in OPPOSITES:
            asked_traits.add(OPPOSITES[best_trait])

        cat, val = best_trait
        last_category = cat
        question_count += 1

        q_text = generate_neural_question(cat, val)

        while True:
            raw_resp = input(f"[Q{question_count}] {q_text}\n  (y/p/k/d/n): ").strip().lower()
            if raw_resp in CONFIDENCE_MAP:
                p_yes = CONFIDENCE_MAP[raw_resp]
                break
            print("Invalid input. Choose: y, p, k, d, or n.")

        for animal in ALL_ANIMALS:
            has_trait = best_trait in get_animal_traits(ANIMAL_DB[animal])
            likelihood = p_yes if has_trait else (1.0 - p_yes)
            probs[animal] *= likelihood

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
