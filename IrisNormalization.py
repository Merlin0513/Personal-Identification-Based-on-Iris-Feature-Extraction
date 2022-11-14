import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Given an image, center of the pupil and iris, radius of the pupil and iris and 
# height and width of the expected output. We return a standardized dimension image
# projecting the original iris in a Cartesian coordinate system

def IrisNormalization(image, height, width, circles_pupil, circles_iris):     
    # Values of theta from 0 to 2*pi
    theta_list = np.arange(0, 2 * np.pi, 2 * np.pi / width) 
    # Empty array which will contain the output
    empty = np.zeros((height,width, 3), np.uint8)
    # Coordinates of pupil and iris center
    circle_x = circles_pupil[1]
    circle_y = circles_pupil[0]
    # Radius of pupil and iris
    r_pupil = circles_pupil[2]
    r_iris = circles_iris[2]
    color = [0,0,0]
    for i in range(width):
        for j in range(height):
            theta = theta_list[i]
            r = j / height
            # Compute Xp, Yp, Xi and Yi as in the paper
            Xp = circle_x + r_pupil * np.cos(theta)
            Yp = circle_y + r_pupil * np.sin(theta)
            Xi = circle_x + r_iris * np.cos(theta)
            Yi = circle_y + r_iris * np.sin(theta)
            
            # The matched cartesian coordinates for the polar coordinates
            X = Xp + ( Xi - Xp )*r
            Y = Yp + ( Yi - Yp )*r

            shapes = image.shape
            if X < shapes[0] and Y < shapes[1]: # Checking if X and Y are not out of range
                color = image[int(X)][int(Y)]   # Color of the pixel
            empty[j][i] = color
    return empty
