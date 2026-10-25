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

TRAIT_SPEECH_MAP = {
    ("diet", "herbivore or plant-eater"): "Does this animal feed mostly on grass, leaves, or plants?",
    ("diet", "carnivore or active predator"): "Does this animal hunt other creatures for meat?",
    ("diet", "omnivore"): "Does this animal eat both plants and meat?",
    ("diet", "insectivore or bug-eater"): "Does this creature primarily eat insects?",
    ("diet", "piscivore"): "Does this animal eat mostly fish?",
    ("diet", "frugivore"): "Does this creature feed mainly on fruits and berries?",
    ("diet", "folivore browser"): "Does this animal browse on tree bark, leaves, and twigs?",
    ("class", "mammal"): "Is this creature a mammal?",
    ("class", "bird"): "Is this animal a bird?",
    ("class", "reptile"): "Is this creature a reptile?",
    ("class", "amphibian"): "Is this creature an amphibian?",
    ("class", "fish"): "Is this animal a fish?",
    ("class", "invertebrate"): "Is this creature an invertebrate (such as an insect, spider, or octopus)?",
    ("limbs", "four legs"): "Does this animal walk on four legs?",
    ("limbs", "cloven hooves"): "Does this animal have split or cloven hooves?",
    ("limbs", "single solid hooves"): "Does this animal have solid hooves like a horse?",
    ("limbs", "sharp retractable claws"): "Does this animal have sharp claws?",
    ("limbs", "two legs and wings"): "Does this animal have wings?",
    ("limbs", "flippers for swimming"): "Does this creature have flippers or paddles for swimming?",
    ("limbs", "pectoral and dorsal fins"): "Does this creature swim using fins?",
    ("limbs", "limbless, moves by slithering"): "Is this animal limbless and moves by slithering?",
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
    ("lifestyle", "domesticated"): "Is this a domesticated animal or pet kept by humans?",
    ("lifestyle", "social or herd-dwelling"): "Does this animal typically live in herds, packs, or social groups?",
    ("lifestyle", "solitary"): "Does this animal live mostly alone rather than in a group?",
    ("lifestyle", "nocturnal or night-active"): "Is this creature active mostly at night?",
    ("lifestyle", "diurnal or active during daylight"): "Is this creature active primarily during the daytime?",
    ("lifestyle", "arboreal tree dweller"): "Does this animal spend most of its time up in trees?",
    ("lifestyle", "semi aquatic"): "Does this creature spend a large part of its life in the water?",
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
    ("head", "tufted ears"): "Does this animal have noticeable tufts of hair on the tips of its ears?",
    ("build", "heavy robust build"): "Does this animal have a very large, heavy, or bulky build?",
    ("build", "tall athletic build"): "Is this animal built tall with a slender or athletic frame?",
    ("special", "produces milk or farmed by humans"): "Is this animal commercially raised on farms for milk, meat, or wool?"
}

def _run_model_inference(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=18, temperature=0.15, top_p=0.8,
            repetition_penalty=1.2, do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

if HAS_SPACES:
    _run_model_inference = spaces.GPU(_run_model_inference)

def generate_natural_question(category, trait_val):
    clean_val = trait_val.replace("_", " ").strip()
    if (category, trait_val) in TRAIT_SPEECH_MAP:
        base_q = TRAIT_SPEECH_MAP[(category, trait_val)]
    elif (category, clean_val) in TRAIT_SPEECH_MAP:
        base_q = TRAIT_SPEECH_MAP[(category, clean_val)]
    else:
        if category == "biome":
            base_q = f"Does this animal live in {clean_val}?"
        elif category == "class":
            base_q = f"Is this animal a {clean_val}?"
        elif category in ["weapons", "covering", "limbs", "head"]:
            base_q = f"Does this animal have {clean_val}?"
        else:
            base_q = f"Is this animal known for {clean_val}?"

    prompt = (
        "Convert the note into one brief, direct Yes/No question for 20 Questions. "
        "Do not offer options. Do not use 'or not'. End with a question mark.\n\n"
        f"Note: {base_q}\n"
        "Question:"
    )
    candidate = _run_model_inference(prompt)
    first_q = candidate.split("\n")[0].split("?")[0].strip()

    bad_markers = [
        " or not", "or sea", "or on land", "high ground or", "fresh water or",
        "either", "yes or no", "is the answer", "long or short", "has either"
    ]
    is_valid_lead = any(first_q.lower().startswith(w) for w in ["is", "does", "can", "would", "has", "do", "are"])
    has_bad_marker = any(b in first_q.lower() for b in bad_markers)

    if is_valid_lead and not has_bad_marker and len(first_q.split()) >= 4:
        return first_q + "?"
    return base_q

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
    gr.Markdown("# ZoogGuess V2: Neural-Bayesian 20 Questions")
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

    restart_btn = gr.Button("Play Again", visible=False)

    demo.load(init_game, outputs=[state_box, q_box, status_box, btn_row, restart_btn])
    restart_btn.click(init_game, outputs=[state_box, q_box, status_box, btn_row, restart_btn])

    for btn, val in [(btn_y, "Yes"), (btn_p, "Probably"), (btn_k, "Don't Know"), (btn_d, "Doubtful"), (btn_n, "No")]:
        btn.click(step_game, inputs=[gr.State(val), state_box], outputs=[state_box, q_box, status_box, btn_row, restart_btn])

if __name__ == "__main__":
    demo.launch()
