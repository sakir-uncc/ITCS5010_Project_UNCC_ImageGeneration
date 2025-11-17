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

METADATA_REWRITTEN = DATASET_ROOT / "metadata_rewritten.csv"
METADATA_PARTIAL = DATASET_ROOT / "metadata_rewritten_partial.csv"
METADATA_ORIGINAL = DATASET_ROOT / "metadata.csv"

# Priority: resume from rewritten -> partial -> original
if METADATA_REWRITTEN.exists():
    METADATA_CSV = METADATA_REWRITTEN
elif METADATA_PARTIAL.exists():
    METADATA_CSV = METADATA_PARTIAL
else:
    METADATA_CSV = METADATA_ORIGINAL

IMG_ROOT = DATASET_ROOT
SANITY_PREVIEW_PATH = DATASET_ROOT / "sanity_preview.csv"

MODEL_PATH = "deepseek-ai/deepseek-vl2-small"
DEVICE = "cuda:0"

print("Using metadata:", METADATA_CSV)
print("Image root:", IMG_ROOT)
print("Device:", DEVICE)
print("CUDA available:", torch.cuda.is_available())


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
device = torch.device(DEVICE)
vl_gpt = vl_gpt.to(device).eval()


# ---------------------------------------------------------------------
# Attributes & mapping
# ---------------------------------------------------------------------
ATTRIBUTE_VOCAB = [
    "trees",
    "grass",
    "path",
    "road",
    "water",
    "building",
    "brick_building",
    "open_plaza",
    "parking_lot",
    "bench",
    "lamp_post",
]

# Priority (lower index = higher importance)
ATTR_PRIORITY = [
    "trees",
    "building",
    "brick_building",
    "path",
    "grass",
    "road",
    "open_plaza",
    "water",
    "parking_lot",
    "bench",
    "lamp_post",
]

# Keywords to detect attributes in original captions
ATTRIBUTE_KEYWORDS = {
    "trees": ["tree", "trees"],
    "grass": ["grass", "lawn"],
    "path": ["walkway", "path", "pathway", "sidewalk"],
    "road": ["road", "street"],
    "parking_lot": ["parking"],
    "water": ["pond", "lake", "water"],
    "building": ["building", "center", "hall", "union"],
    "brick_building": ["brick"],
    "open_plaza": ["plaza", "courtyard", "square"],
    "bench": ["bench"],
    "lamp_post": ["lamp", "light pole", "street light"],
}

ASSISTANT_TAG = "<|Assistant|>:"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def extract_building_name(rel_img_path: str) -> str:
    """
    academic_buildings/Barnard_Hall/000000.jpg -> Barnard Hall
    monuments/Aperture_Sculpture/000000.jpg -> Aperture Sculpture
    """
    parts = rel_img_path.split("/")
    raw = parts[1] if len(parts) >= 2 else Path(rel_img_path).stem
    return raw.replace("_", " ").strip()


def clean_assistant_output(full_text: str) -> str:
    """
    Extract assistant span and strip DeepSeek special tokens.
    """
    if ASSISTANT_TAG in full_text:
        reply = full_text.split(ASSISTANT_TAG, 1)[1]
    else:
        reply = full_text

    reply = reply.strip()
    # Cut off at first '<' (start of any special token)
    lt_pos = reply.find("<")
    if lt_pos != -1:
        reply = reply[:lt_pos]

    reply = re.sub(r"<\|[^>]*\|>", "", reply)
    reply = re.sub(r"<｜[^>]*｜>", "", reply)
    reply = " ".join(reply.split())
    return reply


def parse_attrs_from_deepseek(attr_text: str) -> list:
    """
    From DeepSeek's raw text, keep only tokens in ATTRIBUTE_VOCAB.
    """
    if not attr_text:
        return []
    tokens = [t.strip() for t in attr_text.lower().split(",")]
    attrs = [t for t in tokens if t in ATTRIBUTE_VOCAB]
    # Deduplicate preserving order
    return list(dict.fromkeys(attrs))


def parse_attrs_from_original(orig_caption: str) -> list:
    """
    Infer attributes present in the original caption using simple keyword matching.
    """
    text = orig_caption.lower()
    present = []
    for attr, kws in ATTRIBUTE_KEYWORDS.items():
        if any(kw in text for kw in kws):
            present.append(attr)
    return present


def rank_attributes(attrs: list) -> list:
    """
    Rank attributes by global priority.
    """
    order = {a: i for i, a in enumerate(ATTR_PRIORITY)}
    return sorted(attrs, key=lambda a: order.get(a, 999))


def select_final_attributes(attrs_intersection: list, attrs_deepseek: list) -> list:
    """
    Use up to 3 attributes:
      - Prefer intersection of DeepSeek and original caption
      - Fallback to DeepSeek-only if intersection empty
      - Rank by priority, keep top 3
    """
    base = attrs_intersection if attrs_intersection else attrs_deepseek
    ranked = rank_attributes(base)
    return ranked[:3]


def resolve_building_conflict(attrs: list, rel_img_path: str) -> list:
    """
    If both 'building' and 'brick_building' are present, keep only one
    using a deterministic pseudo-random choice based on MD5 of image path.
    """
    if "building" in attrs and "brick_building" in attrs:
        h = int(hashlib.md5(rel_img_path.encode()).hexdigest(), 16)
        # Even hash -> keep brick_building, remove building
        # Odd hash  -> keep building, remove brick_building
        if h % 2 == 0:
            return [a for a in attrs if a != "building"]
        else:
            return [a for a in attrs if a != "brick_building"]
    return attrs


def build_scene_phrase(attrs: list, rel_img_path: str) -> str:
    """
    Turn 0–3 attributes into a short scene phrase, with some template variety.
    """
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

    templates_two = [
        "near {a1} and {a2}",
        "beside {a1} and {a2}",
        "close to {a1} and {a2}",
        "next to {a1} and {a2}",
    ]
    templates_three = [
        "near {a1}, {a2}, and {a3}",
        "beside {a1}, {a2}, and {a3}",
        "close to {a1}, {a2}, and {a3}",
    ]

    h = int(hashlib.md5(rel_img_path.encode()).hexdigest(), 16)

    if len(attrs) == 1:
        a1 = mapping[attrs[0]]
        return f"near {a1}"

    if len(attrs) == 2:
        a1, a2 = mapping[attrs[0]], mapping[attrs[1]]
        template = templates_two[h % len(templates_two)]
        return template.format(a1=a1, a2=a2)

    # len >= 3
    a1, a2, a3 = mapping[attrs[0]], mapping[attrs[1]], mapping[attrs[2]]
    template = templates_three[h % len(templates_three)]
    return template.format(a1=a1, a2=a2, a3=a3)


def build_caption(building_name: str, scene_phrase: str) -> str:
    """
    Final one-sentence caption.
    """
    return f"{building_name} at UNC Charlotte {scene_phrase}."


def caption_is_valid(caption: str, building_name: str) -> bool:
    """
    Basic sanity checks: building name present, reasonable length, no tokens.
    """
    if not caption:
        return False
    if building_name.lower() not in caption.lower():
        return False
    words = caption.split()
    if len(words) < 6 or len(words) > 18:
        return False
    if "<" in caption or ">" in caption:
        return False
    return True


# ---------------------------------------------------------------------
# Load metadata (resumable)
# ---------------------------------------------------------------------
print("Loading metadata from:", METADATA_CSV)
df = pd.read_csv(METADATA_CSV)

if "caption_rewritten" not in df.columns:
    df["caption_rewritten"] = ""

print("Total rows:", len(df))


# ---------------------------------------------------------------------
# Sanity preview setup – overwrite each run
# ---------------------------------------------------------------------
SANITY_LIMIT = 20
sanity_rows = []
sanity_remaining = SANITY_LIMIT  # always regenerate sanity_preview for this run


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
save_every = 500
max_retries = 2

with torch.no_grad():
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rewriting captions (full v3)"):

        # Skip if caption already rewritten (resumable behavior)
        if isinstance(row.get("caption_rewritten", ""), str) and row["caption_rewritten"].strip():
            continue

        rel_img_path = row["img_path"]
        orig_caption = str(row["caption"])

        img_path = IMG_ROOT / rel_img_path
        if not img_path.is_file():
            print(f"WARNING: missing image at row {idx}: {img_path}")
            continue

        building_name = extract_building_name(rel_img_path)

        # --- DeepSeek attribute extraction ---
        detected_attrs = []
        for attempt in range(max_retries):
            user_prompt = (
                "<image>\n"
                "You are analyzing a UNC Charlotte campus photo.\n"
                "From the following list of visual attributes:\n"
                f"{', '.join(ATTRIBUTE_VOCAB)}\n\n"
                "Return ONLY the attributes that are clearly visible in the image, "
                "as a comma-separated list using these exact tokens. "
                "If you are uncertain about an attribute, DO NOT include it. "
                "Do not add any extra words."
            )

            conversation = [
                {
                    "role": "<|User|>",
                    "content": user_prompt,
                    "images": [str(img_path)],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            pil_images = load_pil_images(conversation)
            inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
            ).to(device)

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

            if detected_attrs:
                break

        orig_attrs = parse_attrs_from_original(orig_caption)
        intersection = list(set(detected_attrs) & set(orig_attrs))

        final_attrs = select_final_attributes(intersection, detected_attrs)
        final_attrs = resolve_building_conflict(final_attrs, rel_img_path)

        # --- Build final caption ---
        if final_attrs:
            scene_phrase = build_scene_phrase(final_attrs, rel_img_path)
            candidate_caption = build_caption(building_name, scene_phrase)
            if caption_is_valid(candidate_caption, building_name):
                final_caption = candidate_caption
            else:
                final_caption = f"{building_name} at UNC Charlotte campus."
        else:
            final_caption = f"{building_name} at UNC Charlotte campus."

        df.at[idx, "caption_rewritten"] = final_caption

        # --- Sanity preview (first 20 rows processed this run) ---
        if sanity_remaining > 0:
            sanity_rows.append({
                "img_path": rel_img_path,
                "original_caption": orig_caption,
                "attributes_detected": ", ".join(detected_attrs),
                "attributes_from_original": ", ".join(orig_attrs),
                "attributes_used": ", ".join(final_attrs),
                "rewritten_caption": final_caption,
            })
            sanity_remaining -= 1

        # --- Periodic partial save ---
        if (idx + 1) % save_every == 0:
            df.to_csv(METADATA_PARTIAL, index=False)
            print(f"Saved partial progress to {METADATA_PARTIAL}")

# Save final sanity preview
if sanity_rows:
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(SANITY_PREVIEW_PATH, index=False)
    print(f"\nSaved sanity preview to {SANITY_PREVIEW_PATH}")

# Final save
df.to_csv(METADATA_REWRITTEN, index=False)
print(f"Done. Final rewritten metadata saved to {METADATA_REWRITTEN}")
