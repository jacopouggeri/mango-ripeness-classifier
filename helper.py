import cv2
import numpy as np
import os
from ultralytics import YOLO
import contextlib

# Load a pretrained YOLO model (recommended for training)
MODEL = YOLO("yolov8n.pt")

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

def process_images(input_folder, output_folder, confidence_threshold=0.6):
    for image_file in os.listdir(input_folder):
        img_path = os.path.join(input_folder, image_file)
        cropped_img, result_flag = crop_mango(MODEL, img_path, confidence_threshold)
        
        if result_flag > 0:
            continue

        final_img = resize_and_pad(cropped_img)
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, final_img)

def crop_mango_direct(model, img, confidence_threshold=0.6):
    with contextlib.redirect_stdout(None):
        results = MODEL.predict(source=img, stream=False)
    
    # Check if any detection was made
    if len(results[0].boxes.xyxy) == 0:
        print(f"Skipping due to no detection.")
        return None, 1

    # Get the index of the bounding box with the highest confidence
    max_conf_index = np.argmax(results[0].boxes.conf.cpu().numpy())
    
    # Check if highest confidence is above threshold
    if results[0].boxes.conf[max_conf_index] < confidence_threshold:
        print(f"Skipping due to low confidence.")
        return None, 2
    
    # Extract the object with the highest confidence
    box = results[0].boxes.xyxy[max_conf_index].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2], 0

def crop_mango(model, img_path, confidence_threshold=0.6):
    img = cv2.imread(img_path)
    return crop_mango_direct(model, img, confidence_threshold)