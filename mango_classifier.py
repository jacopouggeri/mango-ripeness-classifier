import numpy as np
import cv2
import os

# Naive classifier based on color
class NaiveMangoClassifier:
    def __init__(self):
        self.ripe_color = None
        self.raw_color = None

    def train(self, ripe_dir, raw_dir):
        """
        Computes the average color of the ripe and raw mangoes in the training set.
        """
        self.ripe_color = self._compute_avg_color(ripe_dir)
        self.raw_color = self._compute_avg_color(raw_dir)
    
    def predict_direct(self, img):
        """
        Predicts whether the mango in the image is ripe or raw.
        """
        avg_color = np.mean(img, axis=(0, 1))
        
        ripe_dist = np.linalg.norm(avg_color - self.ripe_color)
        raw_dist = np.linalg.norm(avg_color - self.raw_color)
        
        return "ripe" if ripe_dist < raw_dist else "raw"
    
    def predict(self, image_path):
        """
        Predicts whether the mango in the image is ripe or raw.
        """
        img = cv2.imread(image_path)
        return predict_direct(img)

    def _compute_avg_color(self, directory):
        """
        Computes the average color of the images in the given directory.
        """
        color_sum = np.zeros(3)
        count = 0
        
        for image_file in os.listdir(directory):
            img_path = os.path.join(directory, image_file)
            img = cv2.imread(img_path)
            color_sum += np.mean(img, axis=(0, 1))
            count += 1
        
        return color_sum / count
    
    def test_accuracy(self, ripe_test_dir, raw_test_dir):
        """
        Tests the accuracy of the classifier on the given test set.
        """
        total_images = 0
        correct_predictions = 0

        for image_file in os.listdir(ripe_test_dir):
            img_path = os.path.join(ripe_test_dir, image_file)
            if self.predict(img_path) == "ripe":
                correct_predictions += 1
            total_images += 1

        for image_file in os.listdir(raw_test_dir):
            img_path = os.path.join(raw_test_dir, image_file)
            if self.predict(img_path) == "raw":
                correct_predictions += 1
            total_images += 1

        return correct_predictions / total_images