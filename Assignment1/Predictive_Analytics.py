# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
#-------------------- Libraries --------------------
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('module://ipykernel.pylab.backend_inline')
from sklearn import cluster, preprocessing
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import classification_report
from random import seed, randrange

#-------------------- Performance measures --------------------
def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true,y_pred)
    correct = np.trace(cm)
    total = cm.sum()
    acc=0
    if(total>0):
        acc = correct/total
    return acc

def Recall(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true,y_pred)
    classes = cm.shape[0]
    col = cm.sum(axis=1)
    sum1=0
    for i in range(classes):
        divisor = col[i]
        if(divisor!=0):
            sum1+=cm[i][i]/divisor
    re=0
    if(classes>0):
        re = sum1/classes
    return re

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true,y_pred)
    classes = cm.shape[0]
    col = cm.sum(axis=0)
    sum1=0
    for i in range(classes):
        divisor = col[i]
        if(divisor!=0):
            sum1+=cm[i][i]/divisor
    pre=0
    if(classes>0):
        pre = sum1/classes
    return pre

def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    cen = []
    wcs = 0
    for c in range(len(Clusters)):
        cen.append(np.average(Clusters[c], axis = 0))
    for i in range(len(Clusters)):
        for j in range(len(Clusters[i])):
            wcs += np.linalg.norm(Clusters[i][j] - cen[i]) ** 2       
    return(wcs)

def ConfusionMatrix(y_true,y_pred):
    
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """  
    #For 0-10 range
    un = np.unique(y_true)
    y_true=y_true-un[0]
    un = np.unique(y_pred)
    y_pred=y_pred-un[0]
    classes = len(np.unique(np.concatenate((y_true,y_pred))))        
    mul = y_true*classes
    sum1 = mul+y_pred    
    x,bins= np.histogram(sum1,bins=np.arange(classes**2+1))
    mat = x.reshape(classes, classes)
    return(mat)

def KNN(X_train,X_test,Y_train,K):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    dists = np.sum(X_train**2,axis=1) + np.sum(X_test**2, axis=1)[:, np.newaxis] - 2*np.dot(X_test,X_train.T)
    sorted_distance = np.argsort(dists, axis=1)
    top_k_neigh = sorted_distance[:,:K]
    Y_train = Y_train.to_numpy()
    pred_labels = []
    for i in top_k_neigh:
        c, cot = np.unique(Y_train[i], return_counts = True)
        ind = np.argmax(cot)
        pred_labels.append(c[ind])
    pred_labels = np.asarray(pred_labels)
    return pred_labels

def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    predictions = []
    actual = Y_test
    traindata = np.concatenate((X_train, Y_train), axis = 1)
    print("Generating tree:")
    tree = generateTree(traindata)
    print("Generating tree done")
    for row in testdata:
        predictions.append(predict(tree, row))

    print_tree(tree)
    return actual, predictions
    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    #standardizing data 
    sc = preprocessing.StandardScaler()
    std_data = sc.fit_transform(X_train)
    pca_data = pd.DataFrame(std_data)
    
    #Construct Covariance Matrix(C)
    transpose_data = pca_data.T
    C = np.cov(transpose_data)
    
    #Decomposing C to eigen vectors and values
    e_val, e_vec =  np.linalg.eig(C)
    
    #Estimating High-valued Eigen vectors
    eval_sorted = np.argsort(e_val)
    top_evals = eval_sorted[::-1][:N]
    high_valued_vectors = e_vec[:,top_evals]
    PCA = np.empty([pca_data.shape[0], high_valued_vectors.shape[1]])
    i = 0
    for v in high_valued_vectors.T:
            PCA[:,i] = np.dot(pca_data, v.T)
            i += 1
    return PCA
    
def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    #Parameters
    labels = np.zeros(train_y.shape) #Initial assignment of labels
    centroids = X_train[np.random.choice(X_train.shape[0], N, replace=False)] #Random initialization of centroids
    clusters = [[] for x in range(N)]
    max_iter = 200
    for iteration in range(max_iter):
        
        #Calculating the euclidean distance of datapoints to the centroids
        dist = np.zeros((N, X_train.shape[0]))
        for i in range(len(centroids)):
            dist[i] = np.linalg.norm(X_train - centroids[i], axis=1)
        dist = dist.transpose()
        labels = np.argmin(dist, axis=1)
        
        #Clustering of datapoint based on euclidean distance to centroids
        cls_ind = [[] for x in range(N)]
        for l in range(N):
            cls_ind[l] = [i for i,x in enumerate(labels) if x == l]
        clstr = [[] for x in range(N)]
        for i in range(N):
            clstr[i] = [X_train[idx] for idx in cls_ind[i]]  
        clstr = np.array(clstr)
        for i in range(N):
            clstr[i] = np.array(clstr[i])
            centroids[i] = np.mean(clstr[i], axis = 0)
        clusters = clstr
    return clusters

def SklearnSupervisedLearning(X_train,Y_train,X_test,Y_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    N,C= X_test.shape
    result = []
    
    #SVM 
    SVMResult = SVM(X_train,Y_train,X_test)
    print("SVM accuracy : %f" %Accuracy(Y_test,SVMResult))
    result.append(SVMResult)
    
    #Logistiv Regression
    LRResult = LR(X_train,Y_train,X_test)
    print("Logisitic Regression accuracy : %f" %Accuracy(Y_test, LRResult))
    result.append(LRResult)
    
    #Decision Tree
    DTResult = DT(X_train,Y_train,X_test)
    print("Decision Tree accuracy : %f" %Accuracy(Y_test, DTResult))
    result.append(DTResult)
    
    #KNN
    KNNResult = KNN_sk(X_train,Y_train,X_test)
    print("KNN accuracy : %f" %Accuracy(Y_test, KNNResult))
    result.append(KNNResult)
    return result

def SklearnVotingClassifier(X_train,Y_train,X_test, Y_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    ensembleResult = ensemble(X_train,Y_train,X_test)
    # print("Voting Classifier accuracy : %f" %ensembleVC.score(X_test,Y_test))
    print("Voting Classifier accuracy : %f" %Accuracy(Y_test,ensembleResult))
    return ensembleResult


"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""
#-------------------- Part 2 utility functions --------------------
def getGiniIndex(samples, classes):
    total_length = float(0)
    gini = float(0)
    for item in samples:
        total_length += float(len(item))
        
    for sample in samples:
        if(len(sample)==0):
            continue
        sample_len  = float(len(sample))
        score = float(0)
        for i in classes:
            pr = (list(sample[:,-1]).count(i))/sample_len
            score += (pr * pr)
        gini += (1-score) * (sample_len/total_length)
    return gini

def splitData(data, val, index):
    left, right = list(), list()
    for item in data:
        if(item[index] >= val):
            right.append(item)
        else:
            left.append(item)
    return np.array(left), np.array(right)

def splitNode(node):
    left, right = node['sample']
    
    if((left.size==0) or (right.size==0)):
        classes = list()
        if(left.size==0):
            classes = right[:,-1]  
        else:
            classes = left[:,-1]
            
        node['left'] = node['right'] = max(np.unique(classes), key = list(classes).count)
        return
    
    if (len(left)<=1):
        classes = left[:,-1]
        node['left'] =  max(np.unique(classes), key = list(classes).count)
    else:
        node['left'] = evalSplit(left)
        splitNode(node['left'])
    
    if(len(right)<=1):
        classes = right[:,-1]
        node['right'] = max(np.unique(classes), key = list(classes).count)
    else:
        node['right'] = evalSplit(right)
        splitNode(node['right'])

def generateTree(data):
    root = evalSplit(data)
    splitNode(root)
    return root

def evalSplit(data):
    classes = np.unique(data[:,-1])
    splitIndex, splitVal, spiltSample, minscore = float('inf'), float('inf'), None, float('inf')
    
    for index in range(len(data[0])-1):
        for row in data:
            
            samples = list(splitData(data, row[index], index))
            giniScore = getGiniIndex(samples, classes)
            if(giniScore<minscore):
                splitIndex, splitVal, spiltSample, minscore = index, row[index], samples, giniScore
    return {'index':splitIndex, 'val':splitVal, 'sample':spiltSample, 'score':minscore}

def predict(node, row):
    if row[node['index']] < node['val']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def SVM(X_train,Y_train,X_test):
    DIC_svm =  SVC(kernel='linear',gamma='scale')
    DIC_svm.fit(X_train,Y_train)
    return(DIC_svm.predict(X_test))
   
def LR(X_train,Y_train,X_test):
    DIC_logisticRegression = LogisticRegression(solver='lbfgs',multi_class='auto', dual=False, max_iter=8000)
    DIC_logisticRegression.fit(X_train,Y_train)
    return(DIC_logisticRegression.predict(X_test))

def DT(X_train,Y_train,X_test):
    DIC_decisionTreeClassifier = DecisionTreeClassifier()
    DIC_decisionTreeClassifier.fit(X_train,Y_train)
    return(DIC_decisionTreeClassifier.predict(X_test))

def KNN_sk(X_train,Y_train,X_test):
    DIC_KNN = KNeighborsClassifier(n_neighbors=7)
    DIC_KNN.fit(X_train,Y_train)
    KNNResult = DIC_KNN.predict(X_test)
    return DIC_KNN.predict(X_test)

def ensemble(X_train,Y_train,X_test):
    classifierSVC = SVC(kernel='linear')
    classifierLR = LogisticRegression(solver='lbfgs',multi_class='auto', max_iter=8000)
    classifierDT = DecisionTreeClassifier()
    classifierKNN = KNeighborsClassifier(n_neighbors=250)    
    ensembleVC = VotingClassifier(estimators=[("SVM",classifierSVC),("LR",classifierLR),
                                              ("DT",classifierDT),("KNN",classifierKNN)],voting='hard')
    ensembleVC.fit(X_train,Y_train)
    return(ensembleVC.predict(X_test))
    
def subPlotConfusionMatrixElement(ax,confusionMatrix,text):
    ax.matshow(confusionMatrix)
    ax.title.set_text(text)
    ax.xaxis.set_ticks_position('bottom')

def PlotConfusionMatrix(X_train,Y_train,X_test,Y_test):
    SVMResult = SVM(X_train,Y_train,X_test)
    LRResult = LR(X_train,Y_train,X_test)
    DTResult = DT(X_train,Y_train,X_test)
    KNNResult = KNN_sk(X_train,Y_train,X_test)
    ensembleResult = ensemble(X_train,Y_train,X_test)
   
    cmSVM = ConfusionMatrix(Y_test,SVMResult)
    cmLR = ConfusionMatrix(Y_test,LRResult)
    cmDT = ConfusionMatrix(Y_test,DTResult)
    cmKNN = ConfusionMatrix(Y_test,KNNResult)
    cmEnsemble = ConfusionMatrix(Y_test,ensembleResult)
    
    fig, ax = plt.subplots(1, 5, sharex='col', sharey='row',figsize=(15, 4))
    plt.suptitle('Confusion matrix Plots')
    subPlotConfusionMatrixElement(ax[0],cmSVM,"SVM")
    subPlotConfusionMatrixElement(ax[1],cmLR,"Logistic Regression")
    subPlotConfusionMatrixElement(ax[2],cmDT,"Decision Tree")
    subPlotConfusionMatrixElement(ax[3],cmKNN,"K Nearest Neighbors")
    subPlotConfusionMatrixElement(ax[4],cmEnsemble,"Ensemble")
    plt.show()

def PlotGridSearchElement(ax,plt_X,plt_Y,title):
    ax.plot(plt_X,plt_Y)
    ax.title.set_text(title)
    
def GridSearchUtil(X_train, Y_train,function,tuned_parameters):
    gridSearch = GridSearchCV(function, tuned_parameters,cv=5)
    gridSearch.fit(X_train, Y_train) 
    return(gridSearch.cv_results_['mean_test_score'])

def GridSearch(X_train,Y_train,X_test,Y_test):
    fig, ax = plt.subplots(1, 3, sharex='col', sharey='row',figsize=(15, 4))
    plt.suptitle('Grid Search Parameters Plot')
    plt_X = [1,5,10,15]
    tuned_parameters = [{'C': plt_X}]
    plt_Y = GridSearchUtil(X_train, Y_train,SVC(kernel='linear',gamma='scale'),tuned_parameters)
    PlotGridSearchElement(ax[0],plt_X,plt_Y,"SVM")
    plt_X = [5,10,15,20]
    tuned_parameters = [{'max_depth': plt_X}]
    plt_Y = GridSearchUtil(X_train, Y_train,DecisionTreeClassifier(),tuned_parameters)
    PlotGridSearchElement(ax[1],plt_X,plt_Y,"Decision Tree")
    plt_X = [2,10,15,20]
    tuned_parameters = [{'n_neighbors': plt_X}]
    plt_Y = GridSearchUtil(X_train, Y_train,KNeighborsClassifier(),tuned_parameters)
    PlotGridSearchElement(ax[2],plt_X,plt_Y,"K Nearest Neighbors")

#Reading datafile
data = pd.read_csv("data.csv", header = None)
data = data.drop(index=0) #Drops the column names
print("Dataset dimension: ", data.shape[0], "x", data.shape[1]) #dimensions of the data

#Features and label extraction
data_x = data.iloc[:, :-1] #Features
data_y = data.iloc[:, -1] #Labels

#Normalization of data
MinMaxScaler = preprocessing.MinMaxScaler()
data_x = MinMaxScaler.fit_transform(data_x)

#Splittiing data into train (60%), validation (20%) and test (20%)
train_X, cv_t_x, train_y, cv_t_y = train_test_split(data_x, data_y, test_size = 0.2)
cv_x, test_X, cv_y, test_y = train_test_split(cv_t_x, cv_t_y, test_size = 0.5)

print("Training samples: ", train_X.shape[0])
print("Validation samples: ", cv_x.shape[0])
print("Testing samples: ", test_X.shape[0])

#-------------------- Part 2 --------------------
print("-------------------- Part 2 --------------------")
SklearnSupervisedLearning(train_X,train_y,test_X,test_y)
SklearnVotingClassifier(train_X,train_y,test_X,test_y)
PlotConfusionMatrix(train_X,train_y,test_X,test_y)
GridSearch(train_X,train_y,test_X,test_y)

#-------------------- Part 1 --------------------
print("-------------------- Part 1 --------------------")

kr = KNN(train_X,test_X, train_y,15)
print("KNN accuracy : %f" %Accuracy(test_y, kr))

print("Random forest")
RandomForest(train_X,train_y,test_X)

K = len(data_y.unique()) #Number of clusters

#Dimensionality reduction using PCA
print("Dimensionality reduction using PCA")
pca = PCA(train_X,K)

print("KMeans clustering")
clusters = Kmeans(train_X,K)

wcss = WCSS(clusters)
print("Within cluster sum of squares: ", wcss)