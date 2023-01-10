# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 18:53:52 2021

@author: krishna
"""


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
start_time = time.time()

from sklearn import preprocessing 
label = preprocessing.LabelEncoder()
#we use above 2 in encoding the data when obj d.type occurs
df = pd.read_csv("C:/Users/krishna/Desktop/Stars.csv")
#temparature, relative luminosity, relative radius, absolute magnitude,color,spectral class, 
print(df)
print('------------------------------------------------------------')
de= df.head()
print(de)
print('------------------------------------------------------------')
#.head() is used to quickly check if the type of data is correct. By default, first 5 values are displayed
# look for description of data, target and math
print(df.dtypes)
print('------------------------------------------------------------')

vc= df['Type'].value_counts()
print(vc)
print('------------------------------------------------------------')
#prints the number of unique values
check =df.isnull().any()
print(check)
print('------------------------------------------------------------')
#In pandas null means missing. ISNULL() is used to check for missing values in a dataset.
#isnull returns true or false i.e boolean values

df['Color']= label.fit_transform(df['Color'])
df['Spectral_Class']= label.fit_transform(df['Spectral_Class']) 
#since the data type of these 2 cat is obj, we have to encode the and convert them to numbers

print(df['Color'].unique())
print(df['Spectral_Class'].unique())
print('------------------------------------------------------------')
#printing the unique values of spectral class and color. By default it prints in ascending order, i.e color corresponding to no. 5 repeats the least number of times
X = df.drop(['Type'],axis=1)
y = df['Type']
#since the data is being tested for the 'type" classification, we delete this coloumn
fig= plt.figure(figsize=(9,9))
sns.heatmap(X.corr(), annot=True,cmap="YlGnBu")
plt.title("Correlation Matrix of Various types of Stars' Data")
plt.xlabel("Features of various stars")
plt.ylabel("Features of various stars")
print('------------------------------------------------------------')
#print the features that have strong correlation with each other
fig= X.corr()
corr_pairs = fig.unstack()
print(corr_pairs)
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.6,test_size=0.4,random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print('------------------------------------------------------------')
start_time_1 = time.time()
model1 = RandomForestClassifier(n_estimators=100,random_state=0).fit(X_train, y_train)
RF_pred = model1.predict(X_test)
print("time for prediction= %s seconds" % (time.time() - start_time_1))

print('the accuracy',accuracy_score(y_test, RF_pred))

cm = confusion_matrix(y_test, RF_pred)
print(cm)

plt.figure(figsize=(9,9))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(6)
plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5"], rotation=45, size = 5)
plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5"], size = 5)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape

for x in range(width):
    for y in range(height):
        plt.annotate(str(cm[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')
        
print('------------------------------------------------------------')
        
print(classification_report(y_test, RF_pred))
print('------------------------------------------------------------')
print("total execution time of code= %s seconds" % (time.time() - start_time))


