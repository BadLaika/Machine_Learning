# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:30:43 2018

@author: cmerr
"""

#Data Preprocessing
#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
#set a working directory folder
#declare a new variable for data set
dataset = pd.read_csv('')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, :-1].values

#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0) #testsize + train size equals one, dont need to enter both

"""
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""