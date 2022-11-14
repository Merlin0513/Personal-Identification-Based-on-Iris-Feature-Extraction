import numpy as np
import pylab
import cv2
import math
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import distance
from scipy import signal

def m1(x , y, f): #Compute M1 as in the paper
    return np.cos(2*np.pi*f*math.sqrt(x **2 + y**2))

def m2(x , y, f, theta): #Compute M2 as in the paper
    return np.cos(2*np.pi*(x*np.cos(theta)+y*np.sin(theta)))

def gabor1(x, y, dx, dy, f): #Compute Gabor as in the paper with M1
    return (1/(2*math.pi*dx*dy))*np.exp(-0.5*(x**2 / dx**2 + y**2 / dy**2)) * m1(x, y, f)
    
def gabor2(x, y, dx, dy, f, theta): #Compute Gabor as in the paper with M2
    return (1/(2*math.pi*dx*dy))*np.exp(-0.5*(x**2 / dx**2 + y**2 / dy**2)) * m2(x, y, f, theta)

def filters(dx, dy, f): #Create filter for 8x8 small blocks with Gabor and M1
    square = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            square[i,j]=gabor1(j-4,i-4,dx,dy,f)
    return square

def reduction(list1, list2):
    v = []
    for i in range(6):
        for j in range(64):
            # Create 8x8 small blocks
            # For each small block, two feature values are captured. 
            grid1 = list1[i*8:i*8+8,j*8:j*8+8]
            grid2 = list2[i*8:i*8+8,j*8:j*8+8]
            # Mean and the average absolute deviation of the magnitude of 
            # each filtered block defined as in the paper
            absolute = np.absolute(grid1)
            mean = np.mean(absolute)
            v.append(mean)
            std = np.mean(np.absolute(absolute-mean))
            v.append(std)
            # Mean and the average absolute deviation of the magnitude of 
            # each filtered block defined as in the paper
            absolute = np.absolute(grid2)
            mean = np.mean(absolute)
            v.append(mean)
            std = np.mean(np.absolute(absolute-mean))
            v.append(std)
    return v

def FeatureExtraction(image):
    list1=[]
    list2=[]
    # We use the defined spatial filters in two channels to acquire the most 
    # discriminating iris features.
    filter1 = filters(1.5,0.67,3)
    filter2 = filters(1.5,0.67,4.5) 
    # Convert image in gray scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #define a 48x512 region over which the filters are applied
    img = image[:48,:]
    filtered1 = scipy.signal.convolve2d(img,filter1,mode='same')
    filtered2 = scipy.signal.convolve2d(img,filter2,mode='same')
    list1.append(filtered1)
    list2.append(filtered2)
    feature = reduction(filtered1,filtered2)
    return feature
