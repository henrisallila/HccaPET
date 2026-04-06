"""Generate a synthetic test input video for testing the PETSCII converter."""
import cv2
import numpy as np
import sys
import tempfile
import shutil

output = sys.argv[1] if len(sys.argv) > 1 else "test_input.avi"

# Write to temp path first (avoids spaces-in-path issue on Windows)
tmp = tempfile.mktemp(suffix=".avi")
out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"XVID"), 30, (640, 480))
if not out.isOpened():
    print("Error: VideoWriter failed"); sys.exit(1)

for i in range(90):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t = i / 90.0
    for y in range(480):
        frame[y, :] = [int(t * 255), int(y / 480 * 200), int((1 - t) * 255)]
    cx = int(t * 560) + 40
    cv2.circle(frame, (cx, 240), 60, (255, 255, 0), -1)
    cv2.putText(frame, "PETSCII", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    out.write(frame)

out.release()
shutil.move(tmp, output)
print(f"Created: {output}")
