import cv2
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import pandas as pd
from prettytable import PrettyTable
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from ImageEnhancement import ImageEnhancement
from FeatureExtraction import FeatureExtraction
from IrisMatching import IrisMatching
from PerformanceEvaluation import PerformanceEvaluation
import warnings
warnings.filterwarnings("ignore")

# 15 minutes for running on a MacBook Pro
# Change CASIA folder name in the main function (end of this script)

def function(folder_path):
    '''TRAINING'''
    # Training Preprocessing
    feature_vector_train = []         # List containing feature vectors of the training set
    class_train = []                  # Expected class of the training dataset
    for i in range(1,109):
        for j in range(1,4):
            path = folder_path+"/"+'{:03}'.format(i)+'/1/'+'{:03}'.format(i)+'_1_'+str(j)+'.bmp'
            print(path)
            a = IrisLocalization(path)                           # Apply IrisLocalization
            b = IrisNormalization(a[0], 64, 512, a[1], a[2])     # Apply IrisNormalization
            c = ImageEnhancement(b)                              # Apply ImageEnhancement
            d = FeatureExtraction(c)                             # Apply FeatureExtraction
            feature_vector_train.append(d)
            class_train.append(i)
    print("Training data processed.")

    '''TESTING'''
    # Testing Preprocessing
    feature_vector_test = []           # List containing feature vectors of the testing set
    class_test = []                    # Expected class of the testing dataset
    for i in range(1,109):
        for j in range(1,5):
            path = folder_path+"/"+'{:03}'.format(i)+'/2/'+'{:03}'.format(i)+'_2_'+str(j)+'.bmp'
            print(path)
            a = IrisLocalization(path)                           # Apply IrisLocalization
            b = IrisNormalization(a[0], 64, 512, a[1], a[2])     # Apply IrisNormalization
            c = ImageEnhancement(b)                              # Apply ImageEnhancement
            d = FeatureExtraction(c)                             # Apply FeatureExtraction
            feature_vector_test.append(d)
            class_test.append(i)
    print("Testing data processed.")

    components = [15,30,45,60,75,90,105] # Components for the feature vectors
    L = []                               # List containing PerformanceEvaluation for each component
    for component in components:
        predict, minCosine = IrisMatching(feature_vector_train,feature_vector_test,component)
        performance = PerformanceEvaluation(predict,class_test)
        L.append([component,performance])
        print("Performance evaluation made for "+str(component)+" components")

    t1 = PrettyTable(['Similarity measure', 'CRR reduced']) # Table 3 in the paper
    t1.add_row(['L1', L[6][1][0]])
    t1.add_row(['L2', L[6][1][1]])
    t1.add_row(['Cosine', L[6][1][2]])
    print(t1)

    fig, ax =plt.subplots(1,1)
    data=[['L1',L[6][1][0]],
         ['L2',L[6][1][1]],
         ['Cosine',L[6][1][2]]]
    column_labels=["Similarity measure", "CRR reduced"]
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=data,colLabels=column_labels,loc="center")
    fig.savefig('Table_3.png')

    fig = plt.figure(figsize=(10, 5), dpi=150)               # Figure 10 in the paper
    plt.plot(components,[L[i][1][0] for i in range(len(L))])
    plt.xlabel('Components')
    plt.ylabel('CRR')
    plt.title('Recognition results using features of different dimensionality')
    fig.savefig('CRR.png')

    thresh = [0.36+k*0.02 for k in range(10)]                          # Different thresholds
    predict_cosine = [predict[i][2] for i in range(len(predict))]     # Class prediction for cosine method

    match_cosine = []  # List storing 1 if good prediction and 0 if bad prediction
    for i in range(len(predict_cosine)):
        if predict_cosine[i] == class_test[i]:
            match_cosine.append(1)
        else:
            match_cosine.append(0)

    match_cosine_ROC = []           # List storing 1 if minCosine ( minimum cosine value for a testing image)
    for i in range(0,len(thresh)):  # lower than threshold and 0 if greater
        match_ROC=[]
        for j in range(0,len(minCosine)):
            if minCosine[j]<=thresh[i]:
                match_ROC.append(1)
            else:
                match_ROC.append(0)
        match_cosine_ROC.append(match_ROC)


    fmr_all=[]      # Storing False match rate
    fnmr_all=[]     # Storing False non-match rate
    for i in range(len(thresh)):
        false_accept=0
        false_reject=0
        num_1=len([j for j in match_cosine_ROC[i] if j==1])
        num_0=len([j for j in match_cosine_ROC[i] if j==0])

        for k in range(0,len(match_cosine)):
            if match_cosine[k]==0 and match_cosine_ROC[i][k]==1:
                false_accept+=1
            if match_cosine[k]==1 and match_cosine_ROC[i][k]==0:
                false_reject+=1
        fmr=false_accept/num_1
        fnmr=false_reject/num_0
        fmr_all.append(fmr)
        fnmr_all.append(fnmr)

    data = []
    t2 = PrettyTable(['Threshold', 'False match rate (%)', 'False non-match rate (%)']) # Table 4 in the paper
    for i in range(len(thresh)):
        t2.add_row([thresh[i], fmr_all[i], fnmr_all[i]])
        data.append([thresh[i], fmr_all[i], fnmr_all[i]])
    print(t2)

    fig, ax =plt.subplots(1,1)
    column_labels=["Threshold","False match rate (%)","False non-match rate (%)"]
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=data,colLabels=column_labels,loc="center")
    fig.savefig('Table_4.png')

    fig = plt.figure(figsize=(10, 5), dpi=150)  # Figure 11 in the paper
    plt.plot(fnmr_all,fmr_all)
    plt.xlabel('False Match Rate')
    plt.ylabel('False Non-Match Rate')
    plt.title('ROC Curve')
    fig.savefig('ROC.png')




if __name__ == "__main__":
    folder_path = "CASIA Iris Image Database (version 1.0)"  # Folder path on local computer (to change)
    function(folder_path)







