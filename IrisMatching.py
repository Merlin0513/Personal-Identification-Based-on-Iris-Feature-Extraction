import cv2
import numpy as np
import glob
import math
from scipy.spatial import distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

def dim_reduction(list_train, list_test, components):
    
    #get the classes of all training feature vectors
    class_train = []
    for i in range(1,109):
        for j in range(0,3):
            class_train.append(i)
    class_train = np.array(class_train)
    
    #fit the LDA model on training data
    lda = LinearDiscriminantAnalysis(n_components=components)
    lda.fit(list_train, class_train)
    
    #transform the traning data
    reduced_train = lda.transform(list_train)
    
    #transform the testing data
    reduced_test = lda.transform(list_test)
    
    #return transformed training and testing data, and the training classes
    return reduced_train , reduced_test, class_train

def euclidean_distance(row1, row2): # Euclidean distance between two vectors
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return np.sqrt(distance)

def L1_distance(row1, row2):        # L1 distance between two vectors
    distance = 0.0
    for i in range(len(row1)):
        distance += np.abs(row1[i] - row2[i])
    return distance

def L2_distance(row1, row2):        # L2 distance between two vectors
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return distance

def Cosine_distance(row1,row2):     # Cosine distance between two vectors
    return 1 - np.dot(row1,row2)/(euclidean_distance(row1,[0]*len(row1))*euclidean_distance(row2,[0]*len(row2)))


def IrisMatching(list_train, list_test, components):
    reduced_train , reduced_test , class_train = dim_reduction(list_train, list_test, components)
    predict = []         # List storing predicted classes for each testing image for each distance
    minCosine = []       # List storing minimum cosine value for each testing image
    for img_test in reduced_test:
        L1 = []          # For each testing image we compute L1, L2 and Cosine distance with each
        L2 = []          # training image and we compute the class of the index of the minimum value
        Cosine = []      # for each distance
        for img_train in reduced_train:
            L1.append(L1_distance(img_test,img_train))
            L2.append(L2_distance(img_test,img_train))
            Cosine.append(Cosine_distance(img_test,img_train))
        mL1 = class_train[L1.index(min(L1))] # Getting class of training image giving the minimum value for L1 distance
        mL2 = class_train[L2.index(min(L2))] # Getting class of training image giving the minimum value for L2 distance
        mCosine = class_train[Cosine.index(min(Cosine))] # Getting class of training image giving the minimum value for Cosine distance
        predict.append([mL1,mL2,mCosine])
        minCosine.append(min(Cosine))
    return predict, minCosine