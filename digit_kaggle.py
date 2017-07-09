# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 19:12:11 2017

@author: purna
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split  #this is used to split the data into training and the test 
import numpy as np
import pandas as pd
from sklearn import datasets 
import matplotlib.pyplot as plt

digits = datasets.load_digits() 
#print(digits.images.shape)
image_target = list(zip(digits.images,digits.target))                
total_num = len(image_target)
data = digits.images.reshape(total_num,-1)
#print((data.shape))
X_train,X_test,Y_train,Y_test = train_test_split(data,digits.target,test_size = 0.11)
#print((Y_train[0]))
clf = KNeighborsClassifier(n_neighbors = 9,algorithm = 'auto',radius = 1.0)
clf.fit(X_train,Y_train)

print((X_test[0].reshape(1,-1)))
W = clf.predict(X_test[0].reshape(1,-1))
print(W[0])


    



