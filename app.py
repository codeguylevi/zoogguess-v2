import os
import json
import torch
import gradio as gr
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_ID} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None
)
if DEVICE == "cpu":
    model.to("cpu")
model.eval()

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

def determine_stem(category, clean_trait):
    if category == "biome":
        return f"Does your animal make its home in {clean_trait}"
    elif category == "class":
        return f"Is this creature considered a {clean_trait}"
    elif category in ["weapons", "markings", "covering", "head"]:
        return f"Does this animal sport noticeable {clean_trait}"
    elif category == "limbs":
        if any(w in clean_trait for w in ["legs", "feet", "wings", "flippers", "arms"]):
            return f"Does your creature move around using {clean_trait}"
        return f"Does this animal actively {clean_trait}"
    elif category in ["diet", "lifestyle", "build"]:
        if any(w in clean_trait for w in ["predator", "grazer", "omnivore", "hunter", "eater"]):
            return f"Is your creature primarily a {clean_trait}"
        return f"Is this animal known for being {clean_trait}"
    return f"Does your animal have {clean_trait}"

def _run_model_inference(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=18, temperature=0.2, top_p=0.8,
            repetition_penalty=1.3, do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

if HAS_SPACES:
    _run_model_inference = spaces.GPU(_run_model_inference)

def generate_question(category, trait_val):
    clean_trait = trait_val.replace("_", " ").replace("/", " or ")
    stem = determine_stem(category, clean_trait)
    prompt = (
        "Complete the following yes-no question about an animal attribute. "
        "Keep it strictly as a single yes-no question. Do not list alternatives or multiple options.\n\n"
        f"Incomplete question: {stem}"
    )
    text = _run_model_inference(prompt)
    first_clause = text.split("\n")[0].split("?")[0].strip()
    candidate = f"{stem} {first_clause}".strip()
    if not candidate.endswith("?"):
        candidate += "?"
    candidate = " ".join(candidate.split())
    if any(p in candidate.lower() for p in [", or", "what type", "which of", "how many", "rather than", "if so"]):
        return f"{stem}?"
    return candidate

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

def calculate_best_trait(probabilities, asked_traits, last_category):
    trait_weights = defaultdict(float)
    for animal, prob in probabilities.items():
        if prob < 0.001:
            continue
        for t in get_animal_traits(ANIMAL_DB[animal]):
            if t not in asked_traits:
                trait_weights[t] += prob

    best_trait, best_score = None, float("inf")
    for trait, weight in trait_weights.items():
        if not (0.05 < weight < 0.95):
            continue
        cat, _ = trait
        diff = abs(weight - 0.5)
        if cat == last_category and len(probabilities) > 5:
            diff += 0.08
        if diff < best_score:
            best_score, best_trait = diff, trait
    return best_trait

CONFIDENCE_MAP = {"Yes": 0.90, "Probably": 0.75, "Don't Know": 0.50, "Doubtful": 0.25, "No": 0.10}

def init_game():
    n = len(ALL_ANIMALS)
    probs = {a: 1.0 / n for a in ALL_ANIMALS}
    first_trait = calculate_best_trait(probs, set(), None)
    first_q = generate_question(first_trait[0], first_trait[1])
    state = {
        "probs": probs, "asked": [first_trait], "last_cat": first_trait[0],
        "q_count": 1, "done": False, "curr_trait": first_trait
    }
    return state, f"### Question 1\n**{first_q}**", "Game started! Think of an animal.", gr.update(visible=True), gr.update(visible=False)

def step_game(answer, state):
    if not state or state["done"]:
        return state, "Game over! Click 'Play Again' to restart.", "", gr.update(visible=False), gr.update(visible=True)

    probs = state["probs"]
    curr_trait = tuple(state["curr_trait"])
    p_yes = CONFIDENCE_MAP[answer]

    for animal in ALL_ANIMALS:
        has_trait = curr_trait in get_animal_traits(ANIMAL_DB[animal])
        likelihood = p_yes if has_trait else (1.0 - p_yes)
        probs[animal] *= likelihood

    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    state["probs"] = probs

    sorted_candidates = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_animal, top_prob = sorted_candidates[0]
    second_animal, second_prob = sorted_candidates[1]

    top3 = []
    for k, v in sorted_candidates[:3]:
        if v > 0.03:
            name = ANIMAL_DB[k]['display_name']
            top3.append(f"{name} ({v*100:.1f}%)")
    status_text = f"**Current Leading Contenders:** {', '.join(top3)}"

    if top_prob > 0.82 or (top_prob > 0.65 and top_prob > second_prob * 3.5) or state["q_count"] >= 20:
        state["done"] = True
        win_data = ANIMAL_DB[top_animal]
        win_name = win_data['display_name']
        win_desc = win_data['description'][:350]
        res = (
            f"## Deduction Complete in {state['q_count']} Questions!\n"
            f"### You were thinking of: **{win_name}** ({top_prob*100:.1f}% confidence)\n\n"
            f"> {win_desc}..."
        )
        return state, res, status_text, gr.update(visible=False), gr.update(visible=True)

    asked_set = set(tuple(x) for x in state["asked"])
    if curr_trait in OPPOSITES:
        asked_set.add(OPPOSITES[curr_trait])

    best_trait = calculate_best_trait(probs, asked_set, state["last_cat"])
    if not best_trait:
        state["done"] = True
        win_data = ANIMAL_DB[top_animal]
        win_name = win_data['display_name']
        return state, f"## Guess: **{win_name}**", status_text, gr.update(visible=False), gr.update(visible=True)

    state["curr_trait"] = best_trait
    state["asked"].append(best_trait)
    state["last_cat"] = best_trait[0]
    state["q_count"] += 1

    next_q = generate_question(best_trait[0], best_trait[1])
    return state, f"### Question {state['q_count']}\n**{next_q}**", status_text, gr.update(visible=True), gr.update(visible=False)

with gr.Blocks(title="ZoogGuess V2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ZoogGuess V2: Neural-Bayesian 20 Questions")
    gr.Markdown("Think of an animal species. The Bayesian engine calculates optimal information gain while SmolLM2 synthesizes dynamic questions on the fly.")
    
    state_box = gr.State()
    q_box = gr.Markdown()
    status_box = gr.Markdown()

    with gr.Row(visible=True) as btn_row:
        btn_y = gr.Button("Yes", variant="primary")
        btn_p = gr.Button("Probably")
        btn_k = gr.Button("Don't Know")
        btn_d = gr.Button("Doubtful")
        btn_n = gr.Button("No", variant="stop")

    restart_btn = gr.Button("Play Again", visible=False)

    demo.load(init_game, outputs=[state_box, q_box, status_box, btn_row, restart_btn])
    restart_btn.click(init_game, outputs=[state_box, q_box, status_box, btn_row, restart_btn])

    for btn, val in [(btn_y, "Yes"), (btn_p, "Probably"), (btn_k, "Don't Know"), (btn_d, "Doubtful"), (btn_n, "No")]:
        btn.click(step_game, inputs=[gr.State(val), state_box], outputs=[state_box, q_box, status_box, btn_row, restart_btn])

if __name__ == "__main__":
    demo.launch()
