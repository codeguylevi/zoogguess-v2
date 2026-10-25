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

# Complete trait translation map to natural spoken phrases
TRAIT_SPEECH_MAP = {
    # Diets
    ("diet", "herbivore or plant-eater"): "Does this animal feed mostly on grass, leaves, or plants?",
    ("diet", "carnivore or active predator"): "Does this animal hunt other creatures for meat?",
    ("diet", "omnivore"): "Does this animal eat both plants and meat?",
    ("diet", "insectivore or bug-eater"): "Does this creature primarily eat insects?",
    ("diet", "piscivore"): "Does this animal eat mostly fish?",
    ("diet", "frugivore"): "Does this creature feed mainly on fruits and berries?",
    ("diet", "folivore browser"): "Does this animal browse on tree bark, leaves, and twigs?",
    
    # Classes
    ("class", "mammal"): "Is this creature a mammal?",
    ("class", "bird"): "Is this animal a bird?",
    ("class", "reptile"): "Is this creature a reptile?",
    ("class", "amphibian"): "Is this creature an amphibian?",
    ("class", "fish"): "Is this animal a fish?",
    ("class", "invertebrate"): "Is this creature an invertebrate (such as an insect, spider, or octopus)?",

    # Limbs & Locomotion
    ("limbs", "four legs"): "Does this animal walk on four legs?",
    ("limbs", "cloven hooves"): "Does this animal have split or cloven hooves?",
    ("limbs", "single solid hooves"): "Does this animal have solid hooves like a horse?",
    ("limbs", "sharp retractable claws"): "Does this animal have sharp claws?",
    ("limbs", "two legs and wings"): "Does this animal have wings?",
    ("limbs", "flippers for swimming"): "Does this creature have flippers or paddles for swimming?",
    ("limbs", "pectoral and dorsal fins"): "Does this creature swim using fins?",
    ("limbs", "limbless, moves by slithering"): "Is this animal limbless and moves by slithering?",

    # Biomes
    ("biome", "pastures or farmland"): "Is this animal commonly found on farms or agricultural pastures?",
    ("biome", "grasslands"): "Does this animal live in open grasslands or plains?",
    ("biome", "savanna or grassland"): "Does this creature inhabit the open savanna or plains?",
    ("biome", "tropical rainforest or jungle"): "Does this animal make its home in tropical rainforests or jungles?",
    ("biome", "temperate forests or woodlands"): "Does this animal live in temperate forests or woodlands?",
    ("biome", "polar ice or cold tundra"): "Does this animal live in freezing arctic ice or cold tundra?",
    ("biome", "arid desert or scrubland"): "Does this animal inhabit dry desert or scrubland regions?",
    ("biome", "open ocean or coastal waters"): "Does this creature live in the ocean or marine waters?",
    ("biome", "freshwater wetlands, lakes, or rivers"): "Does this animal live in rivers, lakes, or freshwater wetlands?",
    ("biome", "mountainous or alpine highlands"): "Does this creature live in steep mountains or rocky highlands?",

    # Lifestyle
    ("lifestyle", "domesticated"): "Is this a domesticated animal or pet kept by humans?",
    ("lifestyle", "social or herd-dwelling"): "Does this animal typically live in herds, packs, or social groups?",
    ("lifestyle", "solitary"): "Does this animal live mostly alone rather than in a group?",
    ("lifestyle", "nocturnal or night-active"): "Is this creature active mostly at night?",
    ("lifestyle", "diurnal or active during daylight"): "Is this creature active primarily during the daytime?",
    ("lifestyle", "arboreal tree dweller"): "Does this animal spend most of its time up in trees?",
    ("lifestyle", "semi aquatic"): "Does this creature spend a large part of its life in the water?",

    # Coverings & Weapons
    ("covering", "short fur or hair"): "Is this animal covered in a coat of fur or hair?",
    ("covering", "thick fur with spots"): "Does this animal have a spotted fur coat?",
    ("covering", "thick wool or fleece"): "Does this animal have a thick coat of wool or fleece?",
    ("covering", "feathers or plumage"): "Is this creature covered in feathers?",
    ("covering", "scales or scutes"): "Does this animal have scales?",
    ("covering", "exoskeleton or outer shell"): "Does this creature have a hard outer shell or exoskeleton?",
    ("weapons", "horns or blunt defense"): "Does this animal have horns on its head?",
    ("weapons", "permanent horns"): "Does this animal possess permanent horns?",
    ("weapons", "branched antlers"): "Does this animal have branched antlers?",
    ("weapons", "prominent tusks"): "Does this animal have large tusks?",
    ("weapons", "sharp fangs or canine teeth"): "Does this animal have sharp fangs or prominent canine teeth?",
    ("weapons", "venomous bite or stinger"): "Does this creature possess a venomous bite or stinger?",

    # Head & Build & Special
    ("head", "tufted ears"): "Does this animal have noticeable tufts of hair on the tips of its ears?",
    ("build", "heavy robust build"): "Does this animal have a very large, heavy, or bulky build?",
    ("build", "tall athletic build"): "Is this animal built tall with a slender or athletic frame?",
    ("special", "produces milk or farmed by humans"): "Is this animal commercially raised on farms for milk, meat, or wool?"
}

def generate_natural_question(category, trait_val):
    clean_val = trait_val.replace("_", " ").strip()
    
    # 1. Exact canonical question lookup
    if (category, trait_val) in TRAIT_SPEECH_MAP:
        base_q = TRAIT_SPEECH_MAP[(category, trait_val)]
    elif (category, clean_val) in TRAIT_SPEECH_MAP:
        base_q = TRAIT_SPEECH_MAP[(category, clean_val)]
    else:
        # Fallback template
        if category == "biome":
            base_q = f"Does this animal live in {clean_val}?"
        elif category == "class":
            base_q = f"Is this animal a {clean_val}?"
        elif category in ["weapons", "covering", "limbs", "head"]:
            base_q = f"Does this animal have {clean_val}?"
        else:
            base_q = f"Is this animal known for {clean_val}?"

    # 2. Use SmolLM2 for light conversational variety
    prompt = (
        "Convert the note into one brief, direct Yes/No question for 20 Questions. "
        "Do not offer options. Do not use 'or not'. End with a question mark.\n\n"
        f"Note: {base_q}\n"
        "Question:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=18,
            temperature=0.15,
            top_p=0.8,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    candidate = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    first_q = candidate.split("\n")[0].split("?")[0].strip()

    # 3. Validation: reject malformed outputs and fall back to clean canonical question
    bad_markers = [
        " or not", "or sea", "or on land", "high ground or", "fresh water or",
        "either", "yes or no", "is the answer", "long or short", "has either"
    ]
    is_valid_lead = any(first_q.lower().startswith(w) for w in ["is", "does", "can", "would", "has", "do", "are"])
    has_bad_marker = any(b in first_q.lower() for b in bad_markers)

    if is_valid_lead and not has_bad_marker and len(first_q.split()) >= 4:
        return first_q + "?"

    return base_q

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
                v_clean = v.lower().strip()
                if v_clean not in ["none", "unknown", "n/a", "solid coloration", "none of these", ""]:
                    traits.add((cat, v))
        elif isinstance(val, str):
            v_clean = val.lower().strip()
            if v_clean not in ["none", "unknown", "n/a", "solid coloration", "none of these", ""]:
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
        if not (0.08 <= weight <= 0.92):
            continue

        cat, _ = trait
        diff = abs(weight - 0.5)

        # Category lockout: penalize repeating the same category consecutively
        if recent_categories and cat == recent_categories[-1]:
            diff += 0.28
        elif len(recent_categories) >= 2 and cat == recent_categories[-2]:
            diff += 0.14

        if diff < best_score:
            best_score = diff
            best_trait = trait

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
