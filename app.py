import streamlit as st
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

# --- Your imports ---
from SD_finetune.model import build_sd15_model
from SD3_finetune.model import build_sd3_model
from SD_finetune.config import get_default_config as get_sd15_config
from SD3_finetune.config import get_default_config as get_sd3_config
from SD_finetune.inference import generate_images as generate_sd15_images
from SD3_finetune.inference import generate_images as generate_sd3_images
from SD3_finetune.inference import _extract_location_ids_from_prompts
from SD3_finetune.inference import build_token_and_keyword_maps

# -------------------------------------------------------------------
# GLOBAL CONFIG
# -------------------------------------------------------------------
DATASET_ROOT = Path("/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# LOAD LOCATION TOKENS
# -------------------------------------------------------------------
@st.cache_resource
def load_location_maps():
    # Reusing your existing helper
    cfg = get_sd3_config()
    token_to_noun, token_to_classname, keyword_to_token = build_token_and_keyword_maps(cfg["paths"])

    locations = []
    for tok, cls_name in token_to_classname.items():
        slug = "".join([c for c in cls_name.lower() if c.isalnum()])
        noun = token_to_noun.get(tok, "location")
        locations.append({
            "token": tok,
            "class_name": cls_name,
            "slug": slug,
            "noun": noun,
        })

    return locations

# -------------------------------------------------------------------
# LOAD MODELS (CACHED)
# -------------------------------------------------------------------
@st.cache_resource
def load_sd15_finetuned():
    cfg = get_sd15_config()
    paths = cfg["paths"]
    exp = cfg["experiment"]

    ckpt = DATASET_ROOT / "runs/sd15_uncc_fullfinetune/checkpoints/best.pt"
    bundle = build_sd15_model(cfg)
    optimizer, scheduler = bundle, None  # dummy, the loader ignores scheduler
    from SD_finetune.model import load_checkpoint, create_optimizers
    optimizer, scheduler = create_optimizers(bundle, cfg)
    _ = load_checkpoint(bundle, optimizer, scheduler, ckpt)
    bundle.unet.eval()
    bundle.text_encoder.eval()
    return bundle, cfg

@st.cache_resource
def load_sd3_finetuned():
    cfg = get_sd3_config()
    ckpt = DATASET_ROOT / "runs/checkpoint_best.pt"
    bundle = build_sd3_model(cfg)
    from SD3_finetune.model import create_optimizers, load_checkpoint
    optimizer = create_optimizers(cfg, bundle)
    _ = load_checkpoint(cfg, bundle, optimizer, ckpt)
    return bundle, cfg

@st.cache_resource
def load_sd15_base():
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe

@st.cache_resource
def load_sd3_base():
    from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/sd3-medium",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe

# -------------------------------------------------------------------
# IMAGE GENERATION
# -------------------------------------------------------------------
def generate_image(model_name, prompt):

    if model_name == "SD1.5 (Finetuned)":
        bundle, cfg = load_sd15_finetuned()
        imgs = generate_sd15_images(
            bundle,
            prompt=prompt,
            num_images=1,
            steps=40,
        )
        return TF.to_pil_image(imgs[0])

    if model_name == "SD3 (Finetuned)":
        bundle, cfg = load_sd3_finetuned()
        imgs = generate_sd3_images(
            cfg,
            bundle,
            prompt=prompt,
            num_images=1,
            steps=40,
        )
        return TF.to_pil_image(imgs[0])

    if model_name == "SD1.5 (Base)":
        pipe = load_sd15_base()
        im = pipe(prompt=prompt, num_inference_steps=40).images[0]
        return im

    if model_name == "SD3 (Base)":
        pipe = load_sd3_base()
        im = pipe(prompt=prompt, num_inference_steps=40, guidance_scale=3.5).images[0]
        return im

# -------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------
st.set_page_config(page_title="UNC Charlotte Image Generator", layout="centered")

st.title("📸 UNC Charlotte Campus Image Generator")
st.write("Generate images conditioned on campus locations + attributes.")

# Load location options
locations = load_location_maps()
loc_options = [f"{loc['class_name']} ({loc['token']})" for loc in locations]

chosen_loc = st.selectbox("Choose a location:", loc_options)

# Extract selected info
chosen = locations[loc_options.index(chosen_loc)]
token = chosen["token"]
noun = chosen["noun"]
cls_name = chosen["class_name"]

model_name = st.selectbox(
    "Choose model:",
    ["SD1.5 (Finetuned)", "SD3 (Finetuned)", "SD1.5 (Base)", "SD3 (Base)"]
)

st.write("### Optional Attributes (0–3)")
attr1 = st.text_input("Attribute 1", "")
attr2 = st.text_input("Attribute 2", "")
attr3 = st.text_input("Attribute 3", "")

if st.button("Generate Image"):
    with st.spinner("Generating..."):

        # Build extended prompt
        attr_list = [a.strip() for a in [attr1, attr2, attr3] if a.strip() != ""]
        attr_text = ", ".join(attr_list)

        if attr_text:
            prompt = f"a photo of {token} {noun} at UNC Charlotte, {attr_text}"
        else:
            prompt = f"a photo of {token} {noun} at UNC Charlotte"

        img = generate_image(model_name, prompt)
        st.image(img, caption=f"{model_name} — {cls_name}", use_column_width=True)
