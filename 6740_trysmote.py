#from sklearn import processing
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from numpy import loadtxt,where
from scipy import interp
import numpy as np
import pandas as pd
import math
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import itertools

# df1 = pd.read_csv('/Users/Antares/Documents/Rstudio/bank-full-allnum-train.csv')
# df1.info()
# X, y = df1.drop(['y'], axis=1), df1.y
# colNames = list(X.columns.values)
# sm = SMOTE(kind = 'borderline2')
# X40, y40 = sm.fit_sample(X, y)
# X40 = pd.DataFrame(X40, columns = colNames)
# y40 = pd.DataFrame(y40, columns = ['y'])
# sampleSmote = pd.concat([y40, X40],axis = 1)
# sampleSmote.to_csv('/Users/Antares/Documents/sampleSmoteB2.csv', sep=',')

sampleSmote = pd.read_csv('/Users/Antares/Documents/Rstudio/feature_engin/sampleSmote2.csv')
sampleSmote.info()
df1 = pd.read_csv('/Users/Antares/Documents/Rstudio/feature_engin/bank-full-allnum-train.csv')
df1.info()
dataIndex = df1.as_matrix()
dataset = sampleSmote.as_matrix()
#print(isinstance(dataset,np.ndarray)) True
#print(len(dataset[0])) == 20
length = len(dataset)
for col in range(14):
	for k in range(34421,61117):
		dataset[k,col] = round(dataset[k,col])
#### set year as int
for k in range(34421,61117):
	dataset[k,18] = round(dataset[k,18])

print(dataset[34420,:])
print(dataset[34421,:])
print(dataset[35800,:])
print(dataset[47504,:])
print(dataset[53100,:])
print(dataset[52741,:])
print(dataset[61117,:])
print(dataset[1,0])
##### deal with float
# floatList = [16,17,18]
floatList = [15,16,17,]
for col in floatList:
	uniqueValue = list(set(dataIndex[:,col]))
	for k in range(34421,61117):
		dist = list()
		for val in uniqueValue:
			distV = abs(dataset[k,col]-val)
			dist.append(distV)
		nearIndex = np.argmin(dist)
		dataset[k,col] = uniqueValue[nearIndex]

dataset = pd.DataFrame(dataset)
col = list(dataset.columns.values)
colList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18]
for col in colList:
	dataset[col] = dataset[col].astype(int)

dataset.to_csv('/Users/Antares/Documents/sampleSmoteClean.csv', sep=',')







