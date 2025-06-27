import cv2
import numpy as np
from mss import mss
import time

# Load OpenCV's pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Set up screen capture
sct = mss()
monitor = {"top": 650, "left": 3050, "width": 500, "height": 900}  # Adjust as needed

# Frame skipping: process every nth frame for detection
frame_skip = 5
count = 0

# Variable to store the last annotated frame
last_annotated_frame = None

print("Starting facial detection livestream. Press 'q' to quit.")

prev_time = time.time()

while True:
    # Capture screen frame
    capture_start = time.time()
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    capture_time = time.time() - capture_start

    # Run detection every nth frame
    if count % frame_skip == 0:
        inference_start = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        annotated_frame = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        last_annotated_frame = annotated_frame
        inference_time = time.time() - inference_start
    else:
        annotated_frame = frame.copy() if last_annotated_frame is not None else frame
        inference_time = 0.0

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Add overlay text
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Capture: {capture_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Inference: {inference_time:.3f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the result
    cv2.imshow("Face Detection Live", annotated_frame)
    print(f"Capture: {capture_time:.3f}s, Inference: {inference_time:.3f}s, FPS: {fps:.1f}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    count += 1

cv2.destroyAllWindows()
