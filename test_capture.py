import cv2
import numpy as np
from mss import mss

sct = mss()
print(mss().monitors)

# Target right half of extended monitor
monitor = {"top": 650, "left": 3050, "width": 500, "height": 900}

while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    cv2.imshow("Test Capture", frame)
    print(f"Capturing at {monitor}")
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
