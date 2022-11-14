import cv2
import numpy as np

# Given a low contrast and non uniform brightness image 
# We return a more well-distributed texture image

def ImageEnhancement(image):
    output = []         
    for line in image:
        line = line.astype(np.uint8)  # Equalization of the contrast and brightness
        im = cv2.equalizeHist(line)
        output.append(im)
    return np.array([output[i] for i in range(len(output))])