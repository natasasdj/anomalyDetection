import time
import faiss
from sklearn.svm import OneClassSVM

def nnd(features_train, features_test, k):   
    # exact L2 distance 
    # FAISS library
    # input: features_train, features_test
    # output: anomaly score and test runtime
    #k = 1
    d = features_train.shape[1]
    # building index
    index = faiss.IndexFlatL2(d)   
    index.add(features_train)
    # searching nearest neighbour
    start = time.time()
    Dq, Iq = index.search(features_test, k)
    test_runtime = time.time() - start
    anomalyScores = Dq[:,0]
    return anomalyScores, test_runtime

def qnnd(features_train, features_test, k, m, c):
    # approximated L2 distance with product quantization 
    # FAISS library
    # input: features_train, features_test
    # output: anomaly score and test runtime
    #k = 1
    d = features_train.shape[1]
    # building index
    index = faiss.IndexPQ(d, m, c)
    index.train(features_train)
    index.add(features_train)
    # searching nearest neighbour
    start = time.time()
    Dq, Iq = index.search(features_test, k)
    test_runtime = time.time() - start
    anomalyScores = Dq[:,0]
    return anomalyScores, test_runtime

def ocsvm(features_train, features_test):
    # One Class Support Vector Machines
    # fit the model
    ocsvm = OneClassSVM().fit(features_train)
    # predict
    start = time.time()
    anomalyScores = ocsvm.decision_function(features_test)
    test_runtime  = time.time() - start
    return anomalyScores, test_runtime
