import cv2
import os

# --- Input / Output ---
video_path = "videos/vid1.mp4"
output_dir = r"OpenLabeling\main\input"

if os.path.exists(output_dir):
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(file)
else:
    os.makedirs(output_dir)

# --- Config ---
SKIP_FRAMES = 5  # Save every Nth frame

# --- Open video ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second:", fps)

count = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % SKIP_FRAMES == 0:
        filename = f'frame_{saved:04d}.jpg'
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        saved += 1

    count += 1

cap.release()
print(f"Saved {saved} frames (1 every {SKIP_FRAMES}) to: {output_dir}")
