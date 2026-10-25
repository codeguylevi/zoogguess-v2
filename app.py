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

OPPOSITES = {
    ("lifestyle", "solitary"): ("lifestyle", "social_or_pack"),
    ("lifestyle", "social_or_pack"): ("lifestyle", "solitary"),
    ("lifestyle", "nocturnal_night_active"): ("lifestyle", "diurnal_day_active"),
    ("lifestyle", "diurnal_day_active"): ("lifestyle", "nocturnal_night_active"),
    ("diet", "carnivore/predator"): ("diet", "herbivore/grazer"),
    ("diet", "herbivore/grazer"): ("diet", "carnivore/predator")
}

FRIENDLY_TRAIT_MAP = {
    "folivore browser": "feeding primarily on leaves, twigs, and tree bark",
    "frugivore": "eating mostly fruit and berries",
    "insectivore": "eating mostly insects and invertebrates",
    "herbivore/grazer": "eating grass and vegetation",
    "carnivore/predator": "hunting and eating other animals",
    "omnivore": "eating both plants and meat",
    "piscivore": "eating primarily fish",
    "hooves": "walking on hooves",
    "four_legs_quadruped": "walking on all four legs",
    "bipedal_two_legs": "standing and walking on two legs",
    "pectoral_flippers": "having flippers for swimming",
    "prehensile_tail": "having a grasping tail that can hold branches",
    "sagittal crest": "having a prominent bony crest on the top of its head",
    "tufted tail": "having a tail with a tuft of hair on the end",
    "permanent horns": "possessing permanent horns on its head",
    "branched antlers": "possessing branched antlers that shed",
    "blubber fat layer": "having a thick layer of insulating blubber",
    "smooth rubbery hairless skin": "having smooth, hairless, rubbery skin",
    "diurnal_day_active": "being active mostly during daylight hours",
    "nocturnal_night_active": "being active mostly at night",
    "solitary": "living a solitary life rather than in a herd or pack",
    "social_or_pack": "living in herds, packs, or social groups",
    "arboreal tree dweller": "spending most of its life up in the trees",
    "semi aquatic": "spending significant time in the water"
}

def determine_stem(category, clean_trait):
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

def _run_model_inference(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=16, temperature=0.2, top_p=0.8,
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
    if any(p in candidate.lower() for p in [", or", "what type", "which of", "how many", "rather than", "if so", "folivore"]):
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
        if prob < 0.0005:
            continue
        for t in get_animal_traits(ANIMAL_DB[animal]):
            if t not in asked_traits:
                trait_weights[t] += prob

    best_trait, best_score = None, float("inf")
    for trait, weight in trait_weights.items():
        if not (0.04 < weight < 0.96):
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
    return state, f"### Question 1\n**{first_q}**", f"Game started! Database active with {n} species.", gr.update(visible=True), gr.update(visible=False)

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

    if top_prob > 0.82 or (top_prob > 0.65 and top_prob > second_prob * 3.5) or state["q_count"] >= 22:
        state["done"] = True
        win_data = ANIMAL_DB[top_animal]
        win_name = win_data['display_name']
        win_desc = win_data['description'][:350]
        res = (
            f"## 🎉 Deduction Complete in {state['q_count']} Questions!\n"
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
    gr.Markdown("# 🐾 ZoogGuess V2: Neural-Bayesian 20 Questions")
    gr.Markdown("Think of any animal species. The Bayesian engine calculates optimal entropy bisection across 500 species while SmolLM2 synthesizes natural questions.")
    
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
