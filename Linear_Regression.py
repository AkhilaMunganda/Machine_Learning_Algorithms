# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:20:15 2023

@author: AKHILA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df= pd.read_csv('C:/Users/akhil/Documents/Myfolder/AKHILA-NEW/Personal/interview_prep/py-master/ML/1_linear_reg/homeprices.csv')

#Visualizing the data
plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df.area,df.price, color='red', marker= '+')

# Linear regression model
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# Y value
reg.predict([[3300]])
y= reg.predict(df[['area']])
#m value()
reg.coef_
# c value
reg.intercept_


# best fitting line or visual representation of linear regression model
plt.xlabel('area', fontsize = 20) 
plt.ylabel('price', fontsize = 20)
plt.scatter(df.area,df.price, color='red', marker= '+')
plt.plot(df.area, y, color='blue')

#------------------------------------

# creating an array
area = np.array([['1000'],['2000'],['3500'],['4000']])

# converting array to dataframe
df1= pd.DataFrame(area, columns =['area'])

#predicting the price
prices= reg.predict(df1)

# adding the price column to that dataframe.
df1['prices']= prices

# convert it to csv file to desired location
df1.to_csv('C:/Users/akhil/Documents/Myfolder/AKHILA-NEW/Personal/interview_prep/predictprices.csv', index = False)
