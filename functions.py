import cv2
import numpy as np

def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def apply_pixelation(frame, pixel_size=5):
    small_frame = cv2.resize(frame, None, fx=1.0 / pixel_size, fy=1.0 / pixel_size, interpolation=cv2.INTER_NEAREST)
    pixelated_frame = cv2.resize(small_frame, frame.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return pixelated_frame

def apply_mirror(frame):
    frame = cv2.flip(frame, 1)
    half = frame[:frame.shape[0], :frame.shape[1]//2]
    frame[:, frame.shape[1]//2:] = cv2.flip(half, 1)
    return frame

def apply_fisheye(frame, strength=0.5):
    height, width = frame.shape[:2]

    # Define the center of the image
    center_x, center_y = width // 2, height // 2

    # Create a grid of (x, y) coordinates
    y, x = np.indices((height, width))

    # Calculate the distance from the center for each pixel
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Apply the fish-eye distortion
    distorted_x = (x - center_x) * np.exp(-strength * distance / (width / 2)) + center_x
    distorted_y = (y - center_y) * np.exp(-strength * distance / (height / 2)) + center_y

    # Interpolate the distorted coordinates to get the final image
    distorted_frame = cv2.remap(frame, distorted_x.astype(np.float32), distorted_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return distorted_frame

def add_hsv_border(frame, hsv_color):
    # Converte a cor HSV para RGB
    rgb_color = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]

    # Define a espessura da borda
    border_thickness = 1

    # Cria uma cópia do frame
    frame_with_border = frame.copy()

    # Desenha a borda no frame
    frame_with_border = cv2.rectangle(
        frame_with_border,
        (0, 0),
        (frame.shape[1]-1, frame.shape[0]-1),
        rgb_color.tolist(),
        border_thickness,
    )

    return frame_with_border