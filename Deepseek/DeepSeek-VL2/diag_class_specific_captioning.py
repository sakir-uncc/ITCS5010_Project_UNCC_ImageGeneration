import os
import re
import hashlib
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# ---------------------------------------------------------------------
# Paths – adjust if needed
# ---------------------------------------------------------------------
PROJECT_ROOT = Path("/home/sakir/Development/PHD_courses/ITCS5010/Deepseek/DeepSeek-VL2")
DATASET_ROOT = Path("/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1")

METADATA_CSV = DATASET_ROOT / "metadata.csv"
IMG_ROOT = DATASET_ROOT
OUTPUT_PREVIEW = DATASET_ROOT / "sanity_preview_classes.csv"

MODEL_PATH = "deepseek-ai/deepseek-vl2-small"
DEVICE = "cuda:0"


# ---------------------------------------------------------------------
# Load model and processor
# ---------------------------------------------------------------------
print("Loading DeepSeek-VL2 model and processor...")

vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
vl_gpt = vl_gpt.to(DEVICE).eval()


# ---------------------------------------------------------------------
# Attributes & helpers (exact same as main v3 script)
# ---------------------------------------------------------------------
ATTRIBUTE_VOCAB = [
    "trees", "grass", "path", "road", "water", "building",
    "brick_building", "open_plaza", "parking_lot", "bench", "lamp_post",
]

ATTR_PRIORITY = [
    "trees", "building", "brick_building", "path",
    "grass", "road", "open_plaza", "water",
    "parking_lot", "bench", "lamp_post",
]

ATTRIBUTE_KEYWORDS = {
    "trees": ["tree", "trees"],
    "grass": ["grass", "lawn"],
    "path": ["walkway", "path", "pathway", "sidewalk"],
    "road": ["road", "street"],
    "parking_lot": ["parking"],
    "water": ["pond", "lake", "water"],
    "building": ["building", "center", "hall"],
    "brick_building": ["brick"],
    "open_plaza": ["plaza", "courtyard"],
    "bench": ["bench"],
    "lamp_post": ["lamp", "light pole"],
}

ASSISTANT_TAG = "<|Assistant|>:"


def extract_building_name(rel_img_path: str) -> str:
    parts = rel_img_path.split("/")
    raw = parts[1] if len(parts) >= 2 else Path(rel_img_path).stem
    return raw.replace("_", " ").strip()


def clean_assistant_output(full_text: str) -> str:
    if ASSISTANT_TAG in full_text:
        text = full_text.split(ASSISTANT_TAG, 1)[1]
    else:
        text = full_text

    text = text.strip()
    cutoff = text.find("<")
    if cutoff != -1:
        text = text[:cutoff]

    text = re.sub(r"<\|[^>]*\|>", "", text)
    text = re.sub(r"<｜[^>]*｜>", "", text)

    return " ".join(text.split())


def parse_attrs_from_deepseek(txt: str) -> list:
    if not txt:
        return []
    tokens = [t.strip() for t in txt.lower().split(",")]
    return [t for t in tokens if t in ATTRIBUTE_VOCAB]


def parse_attrs_from_original(orig_caption: str) -> list:
    text = orig_caption.lower()
    attrs = []
    for attr, kws in ATTRIBUTE_KEYWORDS.items():
        if any(kw in text for kw in kws):
            attrs.append(attr)
    return attrs

import hashlib

def resolve_building_conflict(attrs: list, rel_img_path: str) -> list:
    """
    If both 'building' and 'brick_building' are present,
    keep only ONE using a deterministic pseudo-random choice
    based on MD5 hash of the image path.
    """
    if "building" in attrs and "brick_building" in attrs:
        h = int(hashlib.md5(rel_img_path.encode()).hexdigest(), 16)
        # Even hash -> keep brick_building.
        # Odd hash  -> keep building.
        if h % 2 == 0:
            return [a for a in attrs if a != "building"]
        else:
            return [a for a in attrs if a != "brick_building"]
    return attrs



def rank_attributes(attrs: list) -> list:
    order = {a: i for i, a in enumerate(ATTR_PRIORITY)}
    return sorted(attrs, key=lambda a: order.get(a, 999))


def select_final_attributes(intersect: list, detected: list) -> list:
    base = intersect if intersect else detected
    ranked = rank_attributes(base)
    return ranked[:3]  # allow up to 3 attributes


def build_scene_phrase(attrs: list, rel_img_path: str) -> str:
    if not attrs:
        return "on the UNC Charlotte campus"

    mapping = {
        "trees": "trees",
        "grass": "a grassy area",
        "path": "a campus walkway",
        "road": "a campus road",
        "water": "a pond",
        "building": "a campus building",
        "brick_building": "a brick building",
        "open_plaza": "an open plaza",
        "parking_lot": "a parking lot",
        "bench": "a bench",
        "lamp_post": "lamp posts",
    }

    t2 = ["near {a1} and {a2}", "beside {a1} and {a2}", "close to {a1} and {a2}", "next to {a1} and {a2}"]
    t3 = ["near {a1}, {a2}, and {a3}", "beside {a1}, {a2}, and {a3}", "close to {a1}, {a2}, and {a3}"]

    h = int(hashlib.md5(rel_img_path.encode()).hexdigest(), 16)

    if len(attrs) == 1:
        return f"near {mapping[attrs[0]]}"

    if len(attrs) == 2:
        a1, a2 = mapping[attrs[0]], mapping[attrs[1]]
        template = t2[h % len(t2)]
        return template.format(a1=a1, a2=a2)

    a1, a2, a3 = (mapping[a] for a in attrs[:3])
    template = t3[h % len(t3)]
    return template.format(a1=a1, a2=a2, a3=a3)


def build_caption(building_name: str, scene_phrase: str) -> str:
    return f"{building_name} at UNC Charlotte {scene_phrase}."


def caption_is_valid(cap: str, name: str) -> bool:
    if name.lower() not in cap.lower():
        return False
    if "<" in cap or ">" in cap:
        return False
    n = len(cap.split())
    return 6 <= n <= 18


# ---------------------------------------------------------------------
# Load metadata
# ---------------------------------------------------------------------
df = pd.read_csv(METADATA_CSV)

# Identify first-example-per-class
df["class_key"] = df["img_path"].apply(lambda p: "/".join(p.split("/")[:2]))

first_rows = df.groupby("class_key").first().reset_index()
print(f"Detected {len(first_rows)} unique classes. Processing one image per class.")

sanity_output = []


# ---------------------------------------------------------------------
# Process each class representative
# ---------------------------------------------------------------------
with torch.no_grad():
    for idx, row in tqdm(first_rows.iterrows(), total=len(first_rows), desc="Class preview"):

        rel_img_path = row["img_path"]
        orig_caption = row["caption"]
        img_path = IMG_ROOT / rel_img_path

        if not img_path.is_file():
            sanity_output.append({
                "img_path": rel_img_path,
                "error": "Missing image"
            })
            continue

        building_name = extract_building_name(rel_img_path)

        # --- Run DeepSeek attribute detection ---
        detected_attrs = []

        user_prompt = (
            "<image>\n"
            "You are analyzing a UNC Charlotte campus photo.\n"
            "From the following list of visual attributes:\n"
            f"{', '.join(ATTRIBUTE_VOCAB)}\n\n"
            "Return ONLY the attributes that are clearly visible in the image, as a comma-separated list. "
            "If you are uncertain about an attribute, DO NOT include it."
        )

        conversation = [
            {"role": "<|User|>", "content": user_prompt, "images": [str(img_path)]},
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        ).to(DEVICE)

        embeds = vl_gpt.prepare_inputs_embeds(**inputs)

        out = vl_gpt.language.generate(
            inputs_embeds=embeds,
            attention_mask=inputs.attention_mask,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        decoded = tokenizer.decode(out[0].cpu().tolist())
        cleaned = clean_assistant_output(decoded)
        detected_attrs = parse_attrs_from_deepseek(cleaned)

        # --- Intersection logic ---
        orig_attrs = parse_attrs_from_original(orig_caption)
        inter = list(set(detected_attrs) & set(orig_attrs))

        final_attrs = select_final_attributes(inter, detected_attrs)

        # Resolve 'building' vs 'brick_building' conflict deterministically
        final_attrs = resolve_building_conflict(final_attrs, rel_img_path)

        # --- Build caption ---
        if final_attrs:
            scene = build_scene_phrase(final_attrs, rel_img_path)
            caption = build_caption(building_name, scene)
            if not caption_is_valid(caption, building_name):
                caption = f"{building_name} at UNC Charlotte campus."
        else:
            caption = f"{building_name} at UNC Charlotte campus."

        # Store result
        sanity_output.append({
            "img_path": rel_img_path,
            "original_caption": orig_caption,
            "attributes_detected": ", ".join(detected_attrs),
            "attributes_from_original": ", ".join(orig_attrs),
            "attributes_used": ", ".join(final_attrs),
            "rewritten_caption": caption,
        })


# ---------------------------------------------------------------------
# Save preview
# ---------------------------------------------------------------------
pd.DataFrame(sanity_output).to_csv(OUTPUT_PREVIEW, index=False)
print("\nSaved class-level preview to:", OUTPUT_PREVIEW)
