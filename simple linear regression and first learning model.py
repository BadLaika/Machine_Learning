# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:53:21 2018

@author: cmerr
"""
#Very first machine learning model wooohoooo


#Data Preprocessing
#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
#set a working directory folder
#declare a new variable for data set
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1/3, random_state = 0) #testsize + train size equals one, dont need to enter both

"""
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""
#fitting simple linear regression model to training set

from sklearn.linear_model import LinearRegression 
#create object of class
regressor = LinearRegression()
#need to fit regressor
#independent var and then targetvalue (depnendent var)
regressor.fit(x_train, y_train)


#create a vector of predicted salaries y_pred of dep var
y_pred = regressor.predict(x_test)
#visualize trainign set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary VS Experience (Training Set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

#visualize test set
plt.scatter(x_test, y_test, color = 'red')
#dont need to change line 55, regressor is already trained, line can stay the same
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary VS Experience (Test Set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

