import cv2
import os

# Put 5–10 raw frames in Image_Correction/test_frames/
frames = [
    "test_frames/frame1.jpg",
    "test_frames/frame2.jpg",
    "test_frames/frame3.jpg",
    "test_frames/frame4.jpg",
    "test_frames/frame5.jpg"
]

os.makedirs("outputs", exist_ok=True)

for i, path in enumerate(frames):
    img = cv2.imread(path)
    if img is None:
        print("Could not load:", path)
        continue

    h, w = img.shape[:2]

    # Draw a reference vertical line at image center
    ref = img.copy()
    cv2.line(ref, (w//2, 0), (w//2, h), (0,255,0), 3)

    outpath = f"outputs/tilt_test_{i}.jpg"
    cv2.imwrite(outpath, ref)
    print("Saved:", outpath)
