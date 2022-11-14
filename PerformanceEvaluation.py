# Given the prediction list ([predict_L1,predict_L2,predict_cosine] for each testing image)
# and the expected classes list of the testing images. 
# We return the CRR for each distance (L1,L2 and Cosine)

def PerformanceEvaluation(predict,class_test):
    L1 = [p[0] for p in predict] # Predict classes for L1 distance
    L2 = [p[1] for p in predict] # Predict classes for L2 distance
    Cosine = [p[2] for p in predict] # Predict classes for Cosine distance
    sl1 = 0
    sl2 = 0
    scosine = 0
    for i in range(len(L1)):     # Comparing prediction and expectation
        if L1[i] == class_test[i]:
            sl1 +=1
        if L2[i] == class_test[i]:
            sl2 +=1
        if Cosine[i] == class_test[i]:
            scosine +=1
    crr_l1 = (sl1/len(L1))*100
    crr_l2 = (sl2/len(L2))*100
    crr_cosine = (scosine/len(Cosine))*100
    return crr_l1,crr_l2,crr_cosine