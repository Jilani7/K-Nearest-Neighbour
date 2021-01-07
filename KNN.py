#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats
import math


# In[2]:


#Read the iris dataset file here
data=pd.read_csv("dataset/iris.data")
data


# In[3]:


data.columns=['SepalLength','SepalWidth','PetalLength','PetalWidth','Class']
data.head()


# In[4]:


data.describe()


# In[5]:


class KNN():
    X_train=0
    y_train=0
    test_size=45
    k = 0
    def __init__(self, k, scalefeatures=False):        
        self.k=k
        scalefeatures = scalefeatures

    def Compute_distances(self,X):
        """
         Compute the distance between each test point in X and each training point
         in self.X_train using a single loop over the test data.
         Input: An num_test x dimension array where each row is a test point.
         A num_test x num_train array where dists[i, j] is the distance
         between the ith test point and the jth training point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sqrt(np.sum((self.X_train - X[i]) ** 2, axis=1))
        return dists
    
    
    def predict_labels(self,X, k = 1):

        """
        Test the trained K-Nearset Neighoubr classifier result on the given examples X


            Input:
            ------
            X: [m x d] a matrix of m  d-dimensional test examples.

            Returns:
            -----------
                pclass: the predicted class for the given set of examples, i.e. to which it belongs
        """
        num_test = X.shape[0]
        pclass = np.zeros(num_test, dtype = 'object')
        for i in range(num_test):
            closest_y = []
            knn_ix = X[i].argsort()[:k]
            closest_y = self.y_train[knn_ix]
            values, counts = np.unique(closest_y, return_counts=True)
            print (values)
            print (counts)
            print(counts == counts.max())
            pclass[i] = values[counts == counts.max()].min()
        return pclass

    
    def predict(self, X):
        dists = self.Compute_distances(X)  
        return self.predict_labels(dists)
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y


# In[6]:


temp1 = data[['SepalLength','SepalWidth','PetalLength','PetalWidth']].dropna()
temp2 = data['Class'].dropna()
X = np.asarray(temp1)
Y = np.asarray(temp2)

print (X)
print (Y)


# In[7]:


def split_data(X, Y, percentage=0.7):
    """
     Split the training data into training and test set according to given percentage... 
    
    Parameters:
    --------
    X: training examples
    Y: training labels
    percentage: split data into train and test accorind to given %
    
    Returns:
    ---------    
    returns four lists as tuple: training data, training labels, test data, test labels 
    """
    
    testp = 1-percentage

    #Split the data into train and test according to given fraction..

    #Creat a list of tuples according to the n-classes where each tuple will 
    # contain the pair of training and test examples for that class...
    #each tuple=(training-examples, training-labels,testing-examples,testing-labels)
    exdata=[]
    #Creat 4 different lists 
    traindata=[]
    trainlabels=[]
    testdata=[]
    testlabels=[]

    classes=np.unique(Y)

    for c in classes:
        # print c
        idx=Y==c
        Yt=Y[idx]
        Xt=X[idx,:]
        nexamples=Xt.shape[0]
        # Generate a random permutation of the indeces
        ridx=np.arange(nexamples) # generate indeces
        np.random.shuffle(ridx)
        ntrainex=round(nexamples*percentage)
        ntestex=nexamples-ntrainex
        ntrainex = int(ntrainex)
        traindata.append(Xt[ridx[:ntrainex],:])
        trainlabels.append(Yt[ridx[:ntrainex]])

        testdata.append(Xt[ridx[ntrainex:],:])
        testlabels.append(Yt[ridx[ntrainex:]])

        #exdata.append((Xt[ridx[:ntrainex],:], Yt[ridx[:ntrainex]], Xt[ridx[ntrainex:],:], Yt[ridx[ntrainex:]]))


    # print traindata,trainlabels
    Xtrain=np.concatenate(traindata)
    Ytrain=np.concatenate(trainlabels)
    Xtest=np.concatenate(testdata)
    Ytest=np.concatenate(testlabels)
    return Xtrain, Ytrain, Xtest, Ytest


# In[8]:


# Spllit Data into train and test
Xtrain,Ytrain,Xtest,Ytest=split_data(X, Y)
print (" Training Data Set Dimensions = ", Xtrain.shape, "Training True Class labels dimensions", Ytrain.shape)   
print (" Test Data Set Dimensions = ", Xtest.shape, "Test True Class labels dimensions", Ytrain.shape)  


# In[9]:


print(Ytrain)


# In[10]:


# Build a 3-nearest neighbour classifier...
x_feature = [0,1,2,3]
knn = KNN(3) 
knn.train(Xtrain[:,x_feature],Ytrain)


# In[11]:


#Check the accuracy on the test set..
labelss = knn.predict(Xtest[:,x_feature])


# In[16]:


print("Accuracy on 3 neighbour classifer is = ", str(np.sum(labelss == Ytest)/float(Ytest.shape[0]) * 100) + "%") 


# In[ ]:





# In[ ]:




