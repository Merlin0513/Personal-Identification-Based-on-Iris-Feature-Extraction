import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def centroid(image): # Compute the centroid of an image (minimum value of projections x and y)
    centroid_x = 0
    centroid_y = 0
    # Image shapes
    width , length = image.shape
    # Initializing X and Y projection to high values in order to compare with lower values
    X = [2000000]*length
    Y = [2000000]*width
    for j in range(width):
        Y[j] = sum(image[j,i] for i in range(length))
        if Y[j] == min(Y):
            centroid_y = j
    for i in range(length):
        X[i] = sum(image[j,i] for j in range(width))
        if X[i] == min(X):
            centroid_x = i  
    return [centroid_x,centroid_y]

def resize(image,centroid): #Resize an image into a 220 square image with centroid as center
    width , length = image.shape
    square = image[max(0,centroid[1]-110):min(centroid[1]+110,width-1),max(0,centroid[0]-110):min(centroid[0]+110,length-1)]
    return square

def circles(image): # Return pupil and iris circles
    im_gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    shapes = im_gray.shape
    im_gray = im_gray[10:shapes[0]-10,10:shapes[1]-10] # Affording border issues
    (thresh, blackAndWhiteImage) = cv2.threshold(im_gray, 70, 255, cv2.THRESH_BINARY)
    C_gray = centroid(blackAndWhiteImage)
    square_200_black = resize(blackAndWhiteImage,C_gray)
    square_200_gray = resize(im_gray,C_gray)
    # Compute pupil circle
    circles_pupil = cv2.HoughCircles(square_200_black,cv2.HOUGH_GRADIENT,1,50,param1=50,param2=10,minRadius=20,maxRadius=100)
    circles_pupil = np.round(circles_pupil[0, :]).astype("int")
    # Compute iris circle for the pupil one
    circles_iris = np.array([[circles_pupil[0][0],circles_pupil[0][1],circles_pupil[0][2]+55]])
    return circles_pupil[0],circles_iris[0],square_200_gray

def IrisLocalization(image):
    circles_pupil,circles_iris,square_200_gray = circles(image)
    # Draw filled circles in white on black background as masks
    mask1 = np.zeros_like(square_200_gray)
    mask1 = cv2.circle(mask1, (circles_pupil[0],circles_pupil[1]), circles_pupil[2], (255,255,255), -1)
    mask2 = np.zeros_like(square_200_gray)
    mask2 = cv2.circle(mask2, (circles_pupil[0],circles_pupil[1]), circles_iris[2], (255,255,255), -1)
    # Subtract masks and make into single channel
    mask = cv2.subtract(mask2, mask1)
    # Put mask into alpha channel of input
    result = cv2.bitwise_and(square_200_gray,square_200_gray,mask = mask)
    return result, circles_pupil, circles_iris