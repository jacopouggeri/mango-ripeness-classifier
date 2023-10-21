import cv2
import numpy as np

def resize_and_pad(cropped_img):
    """
    Resizes the cropped image to fit within 640x640 without distortion.
    Pads the rest with white.
    
    Parameters:
    - cropped_img: numpy array, the cropped image
    
    Returns:
    - final_img: numpy array, the resized and padded image
    """
    max_side = max(cropped_img.shape[0], cropped_img.shape[1])
    scale_factor = 640.0 / max_side
    
    # Resize the image without distortion
    resized_img = cv2.resize(cropped_img, None, fx=scale_factor, fy=scale_factor)

    # Create a blank 640x640 white image
    final_img = np.ones((640, 640, 3), dtype=np.uint8) * 255

    # Get the top-left corner coordinates to place the resized image on the white image
    y_offset = (640 - resized_img.shape[0]) // 2
    x_offset = (640 - resized_img.shape[1]) // 2

    # Place the resized image onto the white image
    final_img[y_offset:y_offset+resized_img.shape[0], x_offset:x_offset+resized_img.shape[1]] = resized_img
    
    return final_img