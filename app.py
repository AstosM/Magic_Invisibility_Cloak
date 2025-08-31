import cv2
import numpy as np
import time
# import gradio as gr

# -----------------------------
# Function for Invisibility Cloak
# -----------------------------
def invisibility(frame, mode="i", custom_img_path=""):
    frame = cv2.flip(frame, 1)  # Flip horizontally
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red cloak range in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    cloak_mask = mask1 + mask2

    # Remove noise
    cloak_mask = cv2.morphologyEx(cloak_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    cloak_mask = cv2.dilate(cloak_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Invert mask
    inverse_mask = cv2.bitwise_not(cloak_mask)

    # Background selection
    if mode == "i":  # Invisible mode (use black background)
        background = np.zeros_like(frame)
    elif mode == "c" and custom_img_path.strip() != "":
        background = cv2.imread(custom_img_path)
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
    else:
        background = np.zeros_like(frame)

    # Extract cloak area from background & non-cloak area from frame
    cloak_area = cv2.bitwise_and(background, background, mask=cloak_mask)
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Combine both
    final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)
    return final_output

# -----------------------------
# Gradio Interface
# -----------------------------
# demo = gr.Interface(
#     fn=invisibility,
#     inputs=[
#         gr.Image(source="webcam", streaming=True, label="Webcam"),  # Webcam input
#         gr.Radio(["i", "c"], label="Mode (i = invisible, c = custom image)", value="i"),
#         gr.Textbox(label="Path to custom image (optional)")
#     ],
#     outputs=gr.Image(),
#     live=True,
#     title="🧙‍♂ Magic Invisibility Cloak",
#     description="Harry Potter–style invisibility cloak using Python + OpenCV. Wear a red cloth to vanish!"
# )

# if __name__ == "__main__":
#     demo.launch()

