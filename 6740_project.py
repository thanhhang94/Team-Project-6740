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
from imblearn.over_sampling import SMOTE



df0 = pd.read_csv('bank-addition-full-year.csv')
df0.info()

#### visualize the dataset. show specific imbalanced between 2 classes
def countplotDef(num):
	plt.title("distribution of subscription of whole dataset")
	for p in num.patches:
		height = p.get_height()
		num.text(p.get_x()+p.get_width()/2.,
			height + 3,
			int(height),
			ha = "center")
	sns.plt.show()

p1 = sns.countplot(df0.y)
countplotDef(p1)

#### transform year to category
df0['year'] = df0['year'].astype(str)
df0.info()

#### check variable correlation
corr = df0.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=True,cbar=True,cmap="coolwarm")
plt.xticks(rotation=90)
plt.show()

#### create a new dataset without those highly correlated variables.(more than 0.9)
df = df0.drop(['emp.var.rate','nr.employed'], axis = 1).copy()
corr = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=True,cbar=True,cmap="coolwarm")
plt.xticks(rotation=90)
plt.show()



df = pd.read_csv('bank-full-train.csv')
p5 = sns.countplot(df.y)
countplotDef(p5)
#### UnderSampling:
group1 = df.loc[df.y == 1].sample(3800).copy()
group0 = df.loc[df.y == 0].sample(5000).copy()
sampleUnder = pd.concat([group1, group0]).sample(8800)
#### write to csv
sampleUnder.to_csv('sampleUnder2.csv', sep=',')
sampleUnder = pd.read_csv('sampleUnder2.csv')

#### over sample the minority class ()
group1 = df.loc[df.y == 1].copy()
while True:
    sampleOver = pd.concat([group1]*6+ [df]).sample(50000)
    print(sampleOver.y.mean())
    if sampleOver.y.mean() > 0.4:
        break
p2 = sns.countplot(sampleOver.y)
countplotDef(p2)
#### write to csv
sampleOver.to_csv('sampleOver2.csv', sep=',')
sampleOver = pd.read_csv('sampleOver2.csv')

#### combine oversample and undersample through smote
from imblearn.over_sampling import SMOTE
df1 = pd.read_csv('bank-full-allnum-train.csv')
X, y = df1.drop(['y'], axis=1), df1.y
colNames = list(X.columns.values)
sm = SMOTE(kind = 'borderline2')
X40, y40 = sm.fit_sample(X, y)
X40 = pd.DataFrame(X40, columns = colNames)
y40 = pd.DataFrame(y40, columns = ['y'])
sampleSmote = pd.concat([y40, X40],axis = 1)
sampleSmote.to_csv('sampleSmote2.csv', sep=',')
sampleSmote = pd.read_csv('sampleSmote2.csv')
p3 = sns.countplot(sampleSmote.y)
countplotDef(p3)

plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(df[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[cn][df.Class == 1], bins=50)
    sns.distplot(df[cn][df.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()

plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(df[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[cn][df.Class == 1], bins=50)
    sns.distplot(df[cn][df.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()



#### read file:
file_name = 'bank-full-train.csv'
df = pd.read_csv(file_name)
df.info()
file_name = 'sampleUnder2.csv'
df1 = pd.read_csv(file_name)
df1.info()

def multi_barplot(dataset,categorical,title):
     # here I have removed left to see who is leaving cpmpany
    fig=plt.subplots(figsize=(12,15))# to define the size of figure
    length=len(categorical) # no of categorical and ordinal variable
    for i,j in itertools.izip_longest(categorical,range(length)): # itertools.zip_longest for to execute the longest loop
        plt.subplot(np.ceil(length/2),2,j+1) # this is to plot the subplots like as 2,2,1 it means 2x2 matrix and graph at 1 
        plt.subplots_adjust(hspace=.5) # to adjust the distance between subplots
        sns.countplot(x=i,data = dataset,hue="y") # To plot the countplot of variable with hue left
        plt.xticks(rotation=90) # to rotate the xticks by 90 such that no xtixks overlap each other
        plt.title(title)
    plt.show()

def barplot(dataset):
    plt.subplot(np.ceil(length/2),2,j+1)
    plt.subplots_adjust(hspace=.5)
    sns.countplot(x='month',data = df,hue="y")
    plt.xticks(rotation=90)
    plt.title("No.of customer who subscribed")
    df.plot(secondary_y=True, label="Comments", legend=True)
    plt.show()

category1=['age','job','education','cons.price.idx','cons.conf.idx','year']
title1 = 'No.of customer who subscribed (original dataset)'
multi_barplot(df,category1,title1)
title2 = 'No.of customer who subscribed (UnderSampling)'
multi_barplot(df1,category1,title2)


barplot

# categorical=['number_project','time_spend_company','Work_accident','promotion_last_5years','sales','salary'] # here I have removed left to see who is leaving cpmpany
# fig=plt.subplots(figsize=(12,15))# to define the size of figure
# length=len(categorical) # no of categorical and ordinal variable
# for i,j in itertools.zip_longest(categorical,range(length)): # itertools.zip_longest for to execute the longest loop
#     plt.subplot(np.ceil(length/2),2,j+1) # this is to plot the subplots like as 2,2,1 it means 2x2 matrix and graph at 1 
#     plt.subplots_adjust(hspace=.5) # to adjust the distance between subplots
#     sns.countplot(x=i,data = Data,hue="left") # To plot the countplot of variable with hue left
#     plt.xticks(rotation=90) # to rotate the xticks by 90 such that no xtixks overlap each other
#     plt.title("No.of employee who left")






