import torch
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


def main():

    # ----------------------------
    # Settings
    # ----------------------------
    model_path = "deepseek-ai/deepseek-vl2-small"
    image_path = "/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1/academic_buildings/barnard_hall/000000.jpg"      # ← put your own image here
    device = "cuda:1"

    print("Loading model and processor...")
    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    model = DeepseekVLV2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,       # we manually to() it below
    )
    model = model.to(device).eval()

    # ----------------------------
    # Build a minimal conversation
    # ----------------------------
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\nDescribe this image in one sentence.",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    print("Loading image...")
    pil_images = load_pil_images(conversation)

    print("Preparing processor inputs...")
    inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt="You are a helpful vision assistant."
    ).to(device)

    print("Encoding inputs...")
    inputs_embeds = model.prepare_inputs_embeds(**inputs)

    print("Generating caption...")
    with torch.inference_mode():
        output = model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=50,
            do_sample=False,
        )

    decoded = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print("\n=== MODEL OUTPUT ===")
    print(decoded)
    print("====================\n")


if __name__ == "__main__":
    main()
