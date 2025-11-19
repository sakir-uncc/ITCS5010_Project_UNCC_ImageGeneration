import re
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATASET_ROOT = Path("/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1")

METADATA_IN = DATASET_ROOT / "metadata_rewritten_dedup_final.csv"

METADATA_WITH_TOKENS = DATASET_ROOT / "metadata_rewritten_dedup_final_with_tokens.csv"
METADATA_TOKENIZED_MINIMAL = DATASET_ROOT / "metadata_tokenized.csv"
TOKENS_TABLE = DATASET_ROOT / "location_tokens.csv"


# ---------------------------------------------------------------------
# CLASS & TOKEN HELPERS
# ---------------------------------------------------------------------

def class_from_img_path(img_path: str) -> str:
    """
    Extract second folder from path.
    Example: 'academic_buildings/Fretwell_hall/000123.jpg' → 'Fretwell_hall'
    """
    parts = img_path.split("/")
    if len(parts) >= 2:
        return parts[1]
    return Path(img_path).parent.name


def normalize_class_name(cls: str) -> str:
    """
    Convert 'Fretwell_hall' -> 'fretwellhall'
    """
    cls = cls.lower()
    cls = re.sub(r"[^a-z0-9]+", "", cls)
    return cls


# ---------------------------------------------------------------------
# CLASS NOUN RULES (category → noun)
# ---------------------------------------------------------------------
CATEGORY_TO_CLASS_NOUN = {
    "academic_buildings": "building",
    "administrative_buildings": "building",
    "athletic_facilities": "facility",
    "student_life_building": "building",
    "monuments": None,   # resolved per subclass below
}

MONUMENT_SUBCLASS_TO_NOUN = {
    "Aperture_Sculpture": "sculpture",
    "Self-Made_Man_Statue": "statue",
    "Hechenbleikner_Lake": "lake",
    "remembrance_memorial": "memorial",
}

def get_class_noun(img_path: str) -> str:
    """
    Determine appropriate class noun for SD training.
    """
    parts = img_path.split("/")
    top = parts[0]
    subclass = parts[1]

    if top == "monuments":
        return MONUMENT_SUBCLASS_TO_NOUN.get(subclass, "monument")

    return CATEGORY_TO_CLASS_NOUN.get(top, "location")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():

    print("\nLoading metadata from:", METADATA_IN)
    df = pd.read_csv(METADATA_IN)

    # Determine source caption column
    if "caption_rewritten" in df.columns:
        src_col = "caption_rewritten"
    elif "caption" in df.columns:
        src_col = "caption"
    else:
        raise ValueError("No 'caption_rewritten' or 'caption' column found in metadata.")

    print(f"Using source caption column: {src_col}")

    # Build class → token mapping
    all_classes = sorted({class_from_img_path(p) for p in df["img_path"].astype(str)})
    class_to_token = {}

    for cls in all_classes:
        norm = normalize_class_name(cls)
        token = f"<loc_{norm}>"
        class_to_token[cls] = token

    print(f"Found {len(all_classes)} classes. Example mappings:")
    for cls in all_classes[:5]:
        print(f"  {cls} → {class_to_token[cls]}")

    # Save token table
    token_df = pd.DataFrame(
        [{"class_name": cls, "token": tok} for cls, tok in sorted(class_to_token.items())]
    )
    token_df.to_csv(TOKENS_TABLE, index=False)
    print("Saved token mapping to:", TOKENS_TABLE)

    # ---------------------------------------------------------
    # GENERATE FINAL TOKENIZED CAPTIONS
    # ---------------------------------------------------------
    tokenized_captions = []

    for _, row in df.iterrows():
        img_path = str(row["img_path"])
        orig_caption = str(row[src_col]).strip()
        cls = class_from_img_path(img_path)
        token = class_to_token[cls]
        class_noun = get_class_noun(img_path)

        # normalize spacing
        orig_caption_clean = " ".join(orig_caption.split())

        # attempt to extract suffix after "at UNC Charlotte"
        marker = "at UNC Charlotte"
        lower_caption = orig_caption_clean.lower()
        lower_marker = marker.lower()

        if lower_marker in lower_caption:
            idx = lower_caption.index(lower_marker)
            suffix = orig_caption_clean[idx + len(marker):].strip()
            scene_suffix = f" {suffix}" if suffix else ""
        else:
            # fallback: preserve entire caption after token
            scene_suffix = f", {orig_caption_clean}"

        # SD-optimized caption template
        final_caption = f"a photo of {token} {class_noun} at UNC Charlotte{scene_suffix}"
        tokenized_captions.append(final_caption)

    df["caption_tokenized"] = tokenized_captions

    # ---------------------------------------------------------
    # SAVE OUTPUTS
    # ---------------------------------------------------------
    df.to_csv(METADATA_WITH_TOKENS, index=False)
    print("Saved extended metadata with tokens:", METADATA_WITH_TOKENS)

    minimal_df = df[["img_path", "caption_tokenized"]]
    minimal_df.to_csv(METADATA_TOKENIZED_MINIMAL, index=False)
    print("Saved minimal tokenized metadata:", METADATA_TOKENIZED_MINIMAL)

    print("\nDone!\n")


if __name__ == "__main__":
    main()
