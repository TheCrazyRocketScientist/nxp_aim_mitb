import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# --- Paths ---
frames_folder = r"OpenLabeling\main\input"
labels_folder = r"OpenLabeling\main\output\YOLO_darknet"
classes_txt = r"OpenLabeling\main\class_list.txt"

# --- Allowed labels (only these will be kept in output) ---
allowed_labels = {
    "qr", "car", "zebra", "teddy bear", "banana",
    "clock", "horse", "potted plant", "cup"
}

# --- Clear old label files ---
if os.path.exists(labels_folder):
    for file in os.listdir(labels_folder):
        file_path = os.path.join(labels_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(labels_folder)

# --- Build sorted class list and ID mapping ---
custom_class_list = sorted(allowed_labels)
custom_class_map = {name: idx for idx, name in enumerate(custom_class_list)}

# --- Save OpenLabeling-compatible class list ---
with open(classes_txt, "w") as f:
    for label in custom_class_list:
        f.write(label + "\n")

# --- Load YOLOv8 model ---
model = YOLO("yolov8s.pt")
model.to("cuda")

# --- Process each frame ---
for filename in tqdm(sorted(os.listdir(frames_folder))):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(frames_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue  # Skip corrupt/missing images

        height, width = image.shape[:2]
        results = model(image, device="cuda")[0]
        label_lines = []

        for box in results.boxes:
            cls_id = int(box.cls.item())
            label = model.names[cls_id]

            if label in allowed_labels:
                custom_id = custom_class_map[label]
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

                # Convert to normalized YOLO format
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height

                label_line = f"{custom_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                label_lines.append(label_line)

        # Save if any allowed detections exist
        if label_lines:
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(labels_folder, label_filename)
            with open(label_path, "w") as lf:
                lf.write("\n".join(label_lines) + "\n")

print(f"YOLO labels saved to: {labels_folder}")
print(f"OpenLabeling class list saved to: {classes_txt}")
