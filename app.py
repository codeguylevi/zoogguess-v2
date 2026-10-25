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

def _run_model_inference(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=20, temperature=0.2, top_p=0.85,
            repetition_penalty=1.2, do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

if HAS_SPACES:
    _run_model_inference = spaces.GPU(_run_model_inference)

def generate_natural_question(category, trait_val):
    clean_val = trait_val.replace("_", " ").strip()
    friendly_val = FRIENDLY_PHRASES.get(clean_val, clean_val)
    template = NATURAL_TEMPLATES.get(category, "Does your animal feature {val}?")
    direct_q = template.format(val=friendly_val)

    prompt = (
        "Rephrase this into a friendly, clear Yes/No question for 20 Questions without changing its meaning. "
        "Keep it strictly under 15 words and end with a question mark:\n\n"
        f"Input: {direct_q}\n"
        "Question:"
    )
    generated = _run_model_inference(prompt)
    first_line = generated.split("\n")[0].split("?")[0].strip()
    if len(first_line) > 5 and any(first_line.lower().startswith(w) for w in ["is", "does", "can", "would", "has", "do", "are"]):
        return first_line + "?"
    return direct_q

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
        if not (0.10 <= weight <= 0.90):
            continue
        cat, _ = trait
        diff = abs(weight - 0.5)
        if cat in recent_categories:
            penalty = 0.25 if (recent_categories and cat == recent_categories[-1]) else 0.12
            diff += penalty
        if diff < best_score:
            best_score = diff
            best_trait = trait

    if not best_trait and trait_weights:
        best_trait = min(trait_weights.keys(), key=lambda t: abs(trait_weights[t] - 0.5))
    return best_trait

CONFIDENCE_MAP = {"Yes": 0.88, "Probably": 0.72, "Don't Know": 0.50, "Doubtful": 0.28, "No": 0.12}

def init_game():
    n = len(ALL_ANIMALS)
    probs = {a: 1.0 / n for a in ALL_ANIMALS}
    first_trait = calculate_best_trait(probs, set(), [])
    first_q = generate_natural_question(first_trait[0], first_trait[1])
    state = {
        "probs": probs, "asked": [first_trait], "recent_cats": [first_trait[0]],
        "q_count": 1, "done": False, "curr_trait": first_trait
    }
    return state, f"### Question 1\n**{first_q}**", f"Game ready with {n} species! Think of an animal to begin.", gr.update(visible=True), gr.update(visible=False)

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

    if top_prob > 0.80 or (top_prob > 0.60 and top_prob > second_prob * 3.5) or state["q_count"] >= 20:
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
    best_trait = calculate_best_trait(probs, asked_set, state["recent_cats"][-2:])
    if not best_trait:
        state["done"] = True
        win_data = ANIMAL_DB[top_animal]
        return state, f"## Guess: **{win_data['display_name']}**", status_text, gr.update(visible=False), gr.update(visible=True)

    state["curr_trait"] = best_trait
    state["asked"].append(best_trait)
    state["recent_cats"].append(best_trait[0])
    state["q_count"] += 1

    next_q = generate_natural_question(best_trait[0], best_trait[1])
    return state, f"### Question {state['q_count']}\n**{next_q}**", status_text, gr.update(visible=True), gr.update(visible=False)

with gr.Blocks(title="ZoogGuess V2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐾 ZoogGuess V2: Neural-Bayesian 20 Questions")
    gr.Markdown("Think of any animal species. The Bayesian engine calculates optimal information gain across 500 species while SmolLM2 synthesizes natural questions.")

    state_box = gr.State()
    q_box = gr.Markdown()
    status_box = gr.Markdown()

    with gr.Row(visible=True) as btn_row:
        btn_y = gr.Button("Yes", variant="primary")
        btn_p = gr.Button("Probably")
        btn_k = gr.Button("Don't Know")
        btn_d = gr.Button("Doubtful")
        btn_n = gr.Button("No", variant="stop")

    restart_btn = gr.Button("🔄 Play Again", visible=False)

    demo.load(init_game, outputs=[state_box, q_box, status_box, btn_row, restart_btn])
    restart_btn.click(init_game, outputs=[state_box, q_box, status_box, btn_row, restart_btn])

    for btn, val in [(btn_y, "Yes"), (btn_p, "Probably"), (btn_k, "Don't Know"), (btn_d, "Doubtful"), (btn_n, "No")]:
        btn.click(step_game, inputs=[gr.State(val), state_box], outputs=[state_box, q_box, status_box, btn_row, restart_btn])

if __name__ == "__main__":
    demo.launch()
