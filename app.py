# %%
import cv2
import numpy as np
import time


# Start webcam
cap = cv2.VideoCapture(0)
time.sleep(3)   # let camera warm up
# # Load custom background image
        # background = cv2.imread("background.jpg")
        # background = cv2.resize(background, (640, 480))
# Capture background (first few frames)
for i in range(30):
    ret, background = cap.read()
background = np.flip(background, axis=1)

# Save output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Invisibility_Magic_cloak.mp4', fourcc, 20.0, (640,480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = np.flip(frame, axis=1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red cloak range
    lower_red1 = np.array([0,120,70])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    # Create mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    cloak_mask = mask1 + mask2

    # Remove noise
    cloak_mask = cv2.morphologyEx(cloak_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    cloak_mask = cv2.dilate(cloak_mask, np.ones((3,3), np.uint8), iterations=1)

    # Invert mask (everything except cloak)
    inverse_mask = cv2.bitwise_not(cloak_mask)

    # Extract cloak area from background & non-cloak area from current frame
    cloak_area = cv2.bitwise_and(background, background, mask=cloak_mask)
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Combine both
    final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

    cv2.imshow("Invisibility Cloak", final_output)
    out.write(final_output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cap.release()
out.release()
cv2.destroyAllWindows()


