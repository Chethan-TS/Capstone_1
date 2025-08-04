from ultralytics import YOLO
import cv2
import os

# --- Paths ---
weights_path = r"E:\Capstone_1\Submission\Submission\Final_customdataset\semi_supervised\final_train_with_refined_labels\weights\best.pt"
input_video_path = r"E:\Capstone_1\Submission\Submission\Final\Test_videos\2.mp4"
output_video_path = r"E:\Capstone_1\Submission\Submission\Final\Test_videos\Custom\2_output.mp4"

# --- Load YOLOv8 model ---
model = YOLO(weights_path)

# --- OpenCV: Read video ---
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# --- Process video frame by frame ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)[0]

    # Draw boxes and labels
    annotated_frame = results.plot()

    # Write frame to output
    out.write(annotated_frame)

# --- Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Object detection video saved to:", output_video_path)
