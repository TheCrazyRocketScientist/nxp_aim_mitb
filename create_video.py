import cv2
import os

# --- Settings ---
frames_folder = "OpenLabeling/main/input"
labels_folder = "OpenLabeling/main/output/YOLO_darknet"
classes_path = "OpenLabeling/main/class_list.txt"
output_video = "assembled_video_with_boxes.mp4"
fps = 30  # Adjust to your original video frame rate

# --- Load class names (optional for labels on boxes) ---
if os.path.exists(classes_path):
    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    class_names = []

# --- Get sorted list of frames ---
images = sorted([
    img for img in os.listdir(frames_folder)
    if img.endswith(".jpg")
])

if not images:
    raise ValueError("No JPG images found in the input folder.")

# --- Get frame size from the first image ---
first_frame_path = os.path.join(frames_folder, images[0])
frame = cv2.imread(first_frame_path)
height, width, _ = frame.shape

# --- Set up video writer ---
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# --- Overlay bboxes and write video ---
for image_name in images:
    frame_path = os.path.join(frames_folder, image_name)
    frame = cv2.imread(frame_path)
    if frame is None:
        continue

    label_file = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(labels_folder, label_file)

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # skip malformed lines

                class_id, x_center, y_center, box_width, box_height = map(float, parts)
                class_id = int(class_id)

                # Convert YOLO normalized coordinates to absolute pixel values
                x1 = int((x_center - box_width / 2) * width)
                y1 = int((y_center - box_height / 2) * height)
                x2 = int((x_center + box_width / 2) * width)
                y2 = int((y_center + box_height / 2) * height)

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Optionally, put class name
                if class_id < len(class_names):
                    cv2.putText(frame, class_names[class_id], (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the annotated frame to video
    video.write(frame)

video.release()
print(f"🎥 Video with bounding boxes saved to: {output_video}")
