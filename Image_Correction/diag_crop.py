import cv2
import numpy as np
import os

# Path to your test frame
input_path = "test_frames/test.jpg"   # <- change filename if needed
output_path = "outputs/crop_test_preview.jpg"

img = cv2.imread(input_path)

if img is None:
    print("ERROR: Could not load image at", input_path)
    exit()

h, w = img.shape[:2]

def center_crop_width(img, crop_ratio):
    new_w = int(w * crop_ratio)
    x1 = (w - new_w) // 2
    x2 = x1 + new_w
    return img[:, x1:x2]

crop_86 = center_crop_width(img, 0.86)
crop_90 = center_crop_width(img, 0.90)

# Resize previews for side-by-side comparison
preview_original = cv2.resize(img, (400, 225))
preview_86 = cv2.resize(crop_86, (400, 225))
preview_90 = cv2.resize(crop_90, (400, 225))

combined = np.hstack([preview_original, preview_86, preview_90])

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

cv2.imwrite(output_path, combined)
print("Saved:", output_path)
