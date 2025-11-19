import cv2
import numpy as np
import os

input_path = "test_frames/test.jpg"
output_path = "outputs/vertical_crop_test.jpg"

img = cv2.imread(input_path)
if img is None:
    print("ERROR loading", input_path)
    exit()

h, w = img.shape[:2]

def center_crop_height(img, crop_ratio):
    new_h = int(h * crop_ratio)
    y1 = (h - new_h) // 2
    y2 = y1 + new_h
    return img[y1:y2, :]

crop_98 = center_crop_height(img, 0.98)
crop_96 = center_crop_height(img, 0.96)

combined = np.vstack([
    cv2.resize(img, (450, 253)),
    cv2.resize(crop_98, (450, 253)),
    cv2.resize(crop_96, (450, 253)),
])

os.makedirs("outputs", exist_ok=True)
cv2.imwrite(output_path, combined)
print("Saved:", output_path)
