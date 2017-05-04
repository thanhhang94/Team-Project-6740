import numpy as np
import pandas as pd

train_data = pd.read_csv('bank-full-train.csv')
test_data = pd.read_csv('bank-full-test.csv')
over_data = pd.read_csv('sampleOver2.csv')
under_data = pd.read_csv('sampleUnder2.csv')
test_smote = pd.read_csv('sampleSmoteTest.csv')
train_smote = pd.read_csv('sampleSmoteClean.csv')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
le=LabelEncoder()

train_data['job']=le.fit_transform(train_data['job'])
train_data['marital']=le.fit_transform(train_data['marital'])
train_data['education']=le.fit_transform(train_data['education'])
train_data['default']=le.fit_transform(train_data['education'])
train_data['housing']=le.fit_transform(train_data['housing'])
train_data['loan']=le.fit_transform(train_data['loan'])
train_data['contact']=le.fit_transform(train_data['contact'])
train_data['month']=le.fit_transform(train_data['month'])
train_data['day_of_week']=le.fit_transform(train_data['day_of_week'])
train_data['poutcome']=le.fit_transform(train_data['poutcome'])
train_data['y']=le.fit_transform(train_data['y'])

test_data['job']=le.fit_transform(test_data['job'])
test_data['marital']=le.fit_transform(test_data['marital'])
test_data['education']=le.fit_transform(test_data['education'])
test_data['default']=le.fit_transform(test_data['education'])
test_data['housing']=le.fit_transform(test_data['housing'])
test_data['loan']=le.fit_transform(test_data['loan'])
test_data['contact']=le.fit_transform(test_data['contact'])
test_data['month']=le.fit_transform(test_data['month'])
test_data['day_of_week']=le.fit_transform(test_data['day_of_week'])
test_data['poutcome']=le.fit_transform(test_data['poutcome'])
test_data['y']=le.fit_transform(test_data['y'])

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as knn
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from __future__ import division

###########All_features##########
def Classification_model(model,train,test,x,y):
  train_x = train.ix[:,x]
  train_y = train.ix[:,y]
  test_x = test.ix[:,x]
  test_y = test.ix[:,y]
  model.fit(train_x,train_y.values.ravel())
  pred=model.predict(test_x)
  accuracy=100*accuracy_score(test_y,pred)
  return accuracy, test_y, pred

All_features = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'year']

Pred_var = ['y']

models = ['KNN','Logistic_Regression','XGBoost']

Classification_models = [knn(n_neighbors=7),LogisticRegression(), XGBClassifier()]
Model_Accuracy = []
Recall = []
Specificity = []
Class0_Err = []
Class1_Err = []
FPR = []
TPR = []
for model in Classification_models:
  Accuracy,actual,predictions=Classification_model(model,train_data,test_data,All_features,Pred_var)
  Model_Accuracy.append(Accuracy)
  conf = confusion_matrix(actual, predictions)
  recall = conf[0][0] / (conf[0][0]+conf[0][1])*100
  spc = conf[1][1]/(conf[1][1]+conf[0][1])*100
  class0_err = conf[0][1]/(conf[0][0]+conf[0][1])*100
  class1_err = conf[1][0]/(conf[1][1]+conf[1][0])*100
  false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
  FPR.append(false_positive_rate)
  TPR.append(true_positive_rate)
  Recall.append(recall)
  Specificity.append(spc)
  Class0_Err.append(class0_err)
  Class1_Err.append(class1_err)

Metrics_with_all_features = pd.DataFrame(
  {"Classification Model" :models,
  "Accuracy with all features":Model_Accuracy,
  "Recall with all features":Recall,
  "Specificity with all features":Specificity
  })

Metrics_with_all_features.sort_values(by="Accuracy with all features",ascending=False).reset_index(drop=True)
  

over_data['job']=le.fit_transform(over_data['job'])
over_data['marital']=le.fit_transform(over_data['marital'])
over_data['education']=le.fit_transform(over_data['education'])
over_data['default']=le.fit_transform(over_data['education'])
over_data['housing']=le.fit_transform(over_data['housing'])
over_data['loan']=le.fit_transform(over_data['loan'])
over_data['contact']=le.fit_transform(over_data['contact'])
over_data['month']=le.fit_transform(over_data['month'])
over_data['day_of_week']=le.fit_transform(over_data['day_of_week'])
over_data['poutcome']=le.fit_transform(over_data['poutcome'])
over_data['y']=le.fit_transform(over_data['y'])

Model_Accuracy_Over = []
Recall_Over = []
Specificity_Over = []
Class0_Err_Over = []
Class1_Err_Over = []
FPR_over = []
TPR_over = []
for model in Classification_models:
  Accuracy_over,actual_over,predictions_over=Classification_model(model,over_data,test_data,All_features,Pred_var)
  Model_Accuracy_Over.append(Accuracy_over)
  conf_over = confusion_matrix(actual_over, predictions_over)
  recall_over = conf_over[0][0] / (conf_over[0][0]+conf_over[0][1])*100
  spc_over = conf_over[1][1]/(conf_over[1][1]+conf_over[0][1])*100
  class0_err_over = conf_over[0][1]/(conf_over[0][0]+conf_over[0][1])*100
  class1_err_over = conf_over[1][0]/(conf_over[1][1]+conf_over[1][0])*100
  false_positive_rate_over, true_positive_rate_over, thresholds_over = roc_curve(actual_over, predictions_over)
  FPR_over.append(false_positive_rate_over)
  TPR_over.append(true_positive_rate_over)
  Recall_Over.append(recall_over)
  Specificity_Over.append(spc_over)
  Class0_Err_Over.append(class0_err_over)
  Class1_Err_Over.append(class1_err_over)

Metrics_with_all_features_Over = pd.DataFrame(
  {"Classification Model_Oversampling":models,
  "Accuracy with all features Oversampling":Model_Accuracy_Over,
  "Recall with all features Oversampling":Recall_Over,
  "Specificity with all features Oversampling":Specificity_Over
  })

Metrics_with_all_features_Over.sort_values(by="Accuracy with all features Oversampling",ascending=False).reset_index(drop=True)

under_data['job']=le.fit_transform(under_data['job'])
under_data['marital']=le.fit_transform(under_data['marital'])
under_data['education']=le.fit_transform(under_data['education'])
under_data['default']=le.fit_transform(under_data['education'])
under_data['housing']=le.fit_transform(under_data['housing'])
under_data['loan']=le.fit_transform(under_data['loan'])
under_data['contact']=le.fit_transform(under_data['contact'])
under_data['month']=le.fit_transform(under_data['month'])
under_data['day_of_week']=le.fit_transform(under_data['day_of_week'])
under_data['poutcome']=le.fit_transform(under_data['poutcome'])
under_data['y']=le.fit_transform(under_data['y'])

Model_Accuracy_Under = []
Recall_Under = []
Specificity_Under = []
Class0_Err_Under = []
Class1_Err_Under = []
FPR_under = []
TPR_under = []
for model in Classification_models:
  Accuracy_under,actual_under,predictions_under=Classification_model(model,under_data,test_data,All_features,Pred_var)
  Model_Accuracy_Under.append(Accuracy_under)
  conf_under = confusion_matrix(actual_under, predictions_under)
  recall_under = conf_under[0][0] / (conf_under[0][0]+conf_under[0][1])*100
  spc_under = conf_under[1][1]/(conf_under[1][1]+conf_under[0][1])*100
  false_positive_rate_under, true_positive_rate_under, thresholds_under = roc_curve(actual_under, predictions_under)
  FPR_under.append(false_positive_rate_under)
  TPR_under.append(true_positive_rate_under)
  Recall_Under.append(recall_under)
  Specificity_Under.append(spc_under)
  class0_err_under = conf_under[0][1]/(conf_under[0][0]+conf_under[0][1])*100
  class1_err_under = conf_under[1][0]/(conf_under[1][1]+conf_under[1][0])*100
  Class0_Err_Under.append(class0_err_under)
  Class1_Err_Under.append(class1_err_under)

Metrics_with_all_features_Under = pd.DataFrame(
  {"Classification Model_Undersampling":models,
  "Accuracy with all features Undersampling":Model_Accuracy_Under,
  "Recall with all features Undersampling":Recall_Under,
  "Specificity with all features Undersampling":Specificity_Under
  })

Metrics_with_all_features_Under.sort_values(by="Accuracy with all features Undersampling",ascending=False).reset_index(drop=True)


Model_Accuracy_Smote = []
Recall_Smote = []
Specificity_Smote = []
Class0_Err_Smote = []
Class1_Err_Smote = []
FPR_smote = []
TPR_smote = []
Total_cost_smote = []
for model in Classification_models:
  Accuracy_sm,actual_sm,predictions_sm=Classification_model(model,train_smote,test_smote,All_features,Pred_var)
  Model_Accuracy_Smote.append(Accuracy_sm)
  conf_sm = confusion_matrix(actual_sm, predictions_sm)
  recall_sm = conf_sm[0][0] / (conf_sm[0][0]+conf_sm[0][1])*100
  spc_sm = conf_sm[1][1]/(conf_sm[1][1]+conf_sm[1][0])*100
  false_positive_rate_smote, true_positive_rate_smote, thresholds_smote = roc_curve(actual_sm, predictions_sm)
  FPR_smote.append(false_positive_rate_smote)
  TPR_smote.append(true_positive_rate_smote)
  Recall_Smote.append(recall_sm)
  Specificity_Smote.append(spc_sm)
  class0_err_smote = conf_sm[0][1]/(conf_sm[0][0]+conf_sm[0][1])*100
  class1_err_smote = conf_sm[1][0]/(conf_sm[1][1]+conf_sm[1][0])*100
  Class0_Err_Smote.append(class0_err_smote)
  Class1_Err_Smote.append(class1_err_smote)
  cost = conf_sm[1][0]*5 + 1*conf_sm[0][1]
  Total_cost_smote.append(cost)

Metrics_with_all_features_Smote = pd.DataFrame(
  {"Classification Model_Smote":models,
  "Accuracy with all features Smote":Model_Accuracy_Smote,
  "Recall with all features Smote":Recall_Smote,
  "Specificity with all features Smote":Specificity_Smote,
  "Total cost with all features Smote":Total_cost_smote
  })

Metrics_with_all_features_Smote.sort_values(by="Accuracy with all features Smote",ascending=False).reset_index(drop=True)

import seaborn

#Logistic_Regression
n=4
fig, ax = plt.subplots()
index = np.arange(n)
bar_width = 0.35
opacity = 0.8

Class0_err_log = []
Class0_err_log.append(Class0_Err[1])
Class0_err_log.append(Class0_Err_Over[1])
Class0_err_log.append(Class0_Err_Under[1])
Class0_err_log.append(Class0_Err_Smote[1])
Class1_err_log = []
Class1_err_log.append(Class1_Err[1])
Class1_err_log.append(Class1_Err_Over[1])
Class1_err_log.append(Class1_Err_Under[1])
Class1_err_log.append(Class0_Err_Smote[1])
rects1 = plt.bar(index, Class0_err_log, bar_width, alpha = opacity, color='r', label='class1_error')
rects2 = plt.bar(index+bar_width, Class1_err_log, bar_width, alpha = opacity, color='b', label='class2_error')
plt.xticks(index + bar_width, ('Original', 'OverSampling', 'UnderSampling', 'Smote'))

plt.xlabel('factor('+'Dataset'+')')
plt.ylabel('percentage')
plt.title('Logistic_Regression')
plt.legend()

plt.tight_layout()
plt.show()

#KNN
n=4
fig, ax = plt.subplots()
index = np.arange(n)
bar_width = 0.35
opacity = 0.8

Class0_err_knn = []
Class0_err_knn.append(Class0_Err[0])
Class0_err_knn.append(Class0_Err_Over[0])
Class0_err_knn.append(Class0_Err_Under[0])
Class0_err_knn.append(Class0_Err_Smote[0])
Class1_err_knn = []
Class1_err_knn.append(Class1_Err[0])
Class1_err_knn.append(Class1_Err_Over[0])
Class1_err_knn.append(Class1_Err_Under[0])
Class1_err_knn.append(Class1_Err_Smote[0])
rects1 = plt.bar(index, Class0_err_knn, bar_width, alpha = opacity, color='r', label='class1_error')
rects2 = plt.bar(index+bar_width, Class1_err_knn, bar_width, alpha = opacity, color='b', label='class2_error')
plt.xticks(index + bar_width, ('Original', 'OverSampling', 'UnderSampling', 'Smote'))

plt.xlabel('factor('+'Dataset'+')')
plt.ylabel('percentage')
plt.title('KNN')
plt.legend()

plt.tight_layout()
plt.show()

#XGBoost
n=4
fig, ax = plt.subplots()
index = np.arange(n)
bar_width = 0.35
opacity = 0.8

Class0_err_xg = []
Class0_err_xg.append(Class0_Err[2])
Class0_err_xg.append(Class0_Err_Over[2])
Class0_err_xg.append(Class0_Err_Under[2])
Class0_err_xg.append(Class0_Err_Smote[2])
Class1_err_xg = []
Class1_err_xg.append(Class1_Err[2])
Class1_err_xg.append(Class1_Err_Over[2])
Class1_err_xg.append(Class1_Err_Under[2])
Class1_err_xg.append(Class1_Err_Smote[2])
rects1 = plt.bar(index, Class0_err_xg, bar_width, alpha = opacity, color='r', label='class1_error')
rects2 = plt.bar(index+bar_width, Class1_err_xg, bar_width, alpha = opacity, color='b', label='class2_error')
plt.xticks(index + bar_width, ('Original', 'OverSampling', 'UnderSampling', 'Smote'))

plt.xlabel('factor('+'Dataset'+')')
plt.ylabel('percentage')
plt.title('XGBoost')
plt.legend()

plt.tight_layout()
plt.show()

#ROC_Log
log_or = []
log_ov = []
log_un = []
log_or.append(FPR[1])
log_ov.append(FPR_over[1])
log_un.append(FPR_under[1])
log_or.append(TPR[1])
log_ov.append(TPR_over[1])
log_un.append(TPR_under[1])

line1=plt.plot(log_or[0],log_or[1], 'b', label='Original_Data')
line2=plt.plot(log_ov[0],log_ov[1], 'r' , label='OverSampling')
line3=plt.plot(log_un[0],log_un[1], 'g' , label='UnderSampling')
plt.legend([line1])
plt.legend([line2])
plt.legend([line3])
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Logistic_Regression')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

#ROC_KNN
knn_or = []
knn_ov = []
knn_un = []
knn_or.append(FPR[0])
knn_ov.append(FPR_over[0])
knn_un.append(FPR_under[0])
knn_or.append(TPR[0])
knn_ov.append(TPR_over[0])
knn_un.append(TPR_under[0])

line4=plt.plot(knn_or[0],knn_or[1], 'b', label='Original_Data')
line5=plt.plot(knn_ov[0],knn_ov[1], 'r' , label='OverSampling')
line6=plt.plot(knn_un[0],knn_un[1], 'g' , label='UnderSampling')
plt.legend([line4])
plt.legend([line5])
plt.legend([line6])
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('KNN')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

#ROC_XG
xg_or = []
xg_ov = []
xg_un = []
xg_or.append(FPR[2])
xg_ov.append(FPR_over[2])
xg_un.append(FPR_under[2])
xg_or.append(TPR[2])
xg_ov.append(TPR_over[2])
xg_un.append(TPR_under[2])

line7=plt.plot(xg_or[0],xg_or[1], 'b', label='Original_Data')
line8=plt.plot(xg_ov[0],xg_ov[1], 'r' , label='OverSampling')
line9=plt.plot(xg_un[0],xg_un[1], 'g' , label='UnderSampling')
plt.legend([line7])
plt.legend([line8])
plt.legend([line9])
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('XGBoost')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
#################Important Features###########
Important_features = ['age', 'job', 'education', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'poutcome', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'year']

Model_Accuracy = []
Recall = []
Specificity = []
Class0_Err = []
Class1_Err = []
FPR = []
TPR = []
Total_cost = []
for model in Classification_models:
  Accuracy,actual,predictions=Classification_model(model,train_data,test_data,Important_features,Pred_var)
  Model_Accuracy.append(Accuracy)
  conf = confusion_matrix(actual, predictions)
  recall = conf[0][0] / (conf[0][0]+conf[0][1])*100
  spc = conf[1][1]/(conf[1][1]+conf[1][0])*100
  class0_err = conf[0][1]/(conf[0][0]+conf[0][1])*100
  class1_err = conf[1][0]/(conf[1][1]+conf[0][1])*100
  cost = conf[1][0]*5 + 1*conf[0][1]
  false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
  Total_cost.append(cost)
  FPR.append(false_positive_rate)
  TPR.append(true_positive_rate)
  Recall.append(recall)
  Specificity.append(spc)
  Class0_Err.append(class0_err)
  Class1_Err.append(class1_err)

Metrics_with_important_features = pd.DataFrame(
  {"Classification Model" :models,
  "Accuracy with important features":Model_Accuracy,
  "Recall with important features":Recall,
  "Specificity with important features":Specificity,
  "Total cost":Total_cost
  })

Metrics_with_important_features.sort_values(by="Accuracy with important features",ascending=False).reset_index(drop=True)

Model_Accuracy_Over = []
Recall_Over = []
Specificity_Over = []
Class0_Err_Over = []
Class1_Err_Over = []
FPR_over = []
TPR_over = []
Total_cost_over = []
for model in Classification_models:
  Accuracy_over,actual_over,predictions_over=Classification_model(model,over_data,test_data,Important_features,Pred_var)
  Model_Accuracy_Over.append(Accuracy_over)
  conf_over = confusion_matrix(actual_over, predictions_over)
  recall_over = conf_over[0][0] / (conf_over[0][0]+conf_over[0][1])*100
  spc_over = conf_over[1][1]/(conf_over[1][1]+conf_over[1][0])*100
  class0_err_over = conf_over[0][1]/(conf_over[0][0]+conf_over[0][1])*100
  class1_err_over = conf_over[1][0]/(conf_over[1][1]+conf_over[0][1])*100
  false_positive_rate_over, true_positive_rate_over, thresholds_over = roc_curve(actual_over, predictions_over)
  cost_over = conf_over[1][0]*5 + 1*conf_over[0][1]
  Total_cost_over.append(cost_over)
  FPR_over.append(false_positive_rate_over)
  TPR_over.append(true_positive_rate_over)
  Recall_Over.append(recall_over)
  Specificity_Over.append(spc_over)
  Class0_Err_Over.append(class0_err_over)
  Class1_Err_Over.append(class1_err_over)

Metrics_with_important_features_Over = pd.DataFrame(
  {"Classification Model_Oversampling":models,
  "Accuracy with important features Oversampling":Model_Accuracy_Over,
  "Recall with important features Oversampling":Recall_Over,
  "Specificity with important features Oversampling":Specificity_Over,
  "Total Cost_Oversampling":Total_cost_over
  })

Metrics_with_important_features_Over.sort_values(by="Accuracy with important features Oversampling",ascending=False).reset_index(drop=True)

Model_Accuracy_Under = []
Recall_Under = []
Specificity_Under = []
Class0_Err_Under = []
Class1_Err_Under = []
FPR_under = []
TPR_under = []
Total_cost_under = []
for model in Classification_models:
  Accuracy_under,actual_under,predictions_under=Classification_model(model,under_data,test_data,Important_features,Pred_var)
  Model_Accuracy_Under.append(Accuracy_under)
  conf_under = confusion_matrix(actual_under, predictions_under)
  recall_under = conf_under[0][0] / (conf_under[0][0]+conf_under[0][1])*100
  spc_under = conf_under[1][1]/(conf_under[1][1]+conf_under[1][0])*100
  false_positive_rate_under, true_positive_rate_under, thresholds_under = roc_curve(actual_under, predictions_under)
  cost_under = conf_under[1][0]*5 + 1*conf_under[0][1]
  Total_cost_under.append(cost_under)
  FPR_under.append(false_positive_rate_under)
  TPR_under.append(true_positive_rate_under)
  Recall_Under.append(recall_under)
  Specificity_Under.append(spc_under)
  class0_err_under = conf_under[0][1]/(conf_under[0][0]+conf_under[0][1])*100
  class1_err_under = conf_under[1][0]/(conf_under[1][1]+conf_under[0][1])*100
  Class0_Err_Under.append(class0_err_under)
  Class1_Err_Under.append(class1_err_under)

Metrics_with_important_features_Under = pd.DataFrame(
  {"Classification Model_Undersampling":models,
  "Accuracy with important features Undersampling":Model_Accuracy_Under,
  "Recall with important features Undersampling":Recall_Under,
  "Specificity with important features Undersampling":Specificity_Under,
  "Total Cost UnderSampling":Total_cost_under
  })

Metrics_with_important_features_Under.sort_values(by="Accuracy with important features Undersampling",ascending=False).reset_index(drop=True)

Model_Accuracy_Smote = []
Recall_Smote = []
Specificity_Smote = []
for model in Classification_models:
  Accuracy_sm,actual_sm,predictions_sm=Classification_model(model,train_smote,test_smote,Important_features,Pred_var)
  Model_Accuracy_Smote.append(Accuracy_sm)
  conf_sm = confusion_matrix(actual_sm, predictions_sm)
  recall_sm = conf_sm[0][0] / (conf_sm[0][0]+conf_sm[0][1])*100
  spc_sm = conf_sm[1][1]/(conf_sm[1][1]+conf_sm[1][0])*100
  Recall_Smote.append(recall_sm)
  Specificity_Smote.append(spc_sm)

Metrics_with_important_features_Smote = pd.DataFrame(
  {"Classification Model_Smote":models,
  "Accuracy with all features Smote":Model_Accuracy_Smote,
  "Recall with all features Smote":Recall_Smote,
  "Specificity with all features Smote":Specificity_Smote
  })

Metrics_with_important_features_Smote.sort_values(by="Accuracy with all features Smote",ascending=False).reset_index(drop=True)


#ROC_Log
log_or = []
log_ov = []
log_un = []
log_or.append(FPR[1])
log_ov.append(FPR_over[1])
log_un.append(FPR_under[1])
log_or.append(TPR[1])
log_ov.append(TPR_over[1])
log_un.append(TPR_under[1])

line1=plt.plot(log_or[0],log_or[1], 'b', label='Original_Data')
line2=plt.plot(log_ov[0],log_ov[1], 'r' , label='OverSampling')
line3=plt.plot(log_un[0],log_un[1], 'g' , label='UnderSampling')
plt.legend([line1])
plt.legend([line2])
plt.legend([line3])
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Logistic_Regression')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

#ROC_KNN
knn_or = []
knn_ov = []
knn_un = []
knn_or.append(FPR[0])
knn_ov.append(FPR_over[0])
knn_un.append(FPR_under[0])
knn_or.append(TPR[0])
knn_ov.append(TPR_over[0])
knn_un.append(TPR_under[0])

line4=plt.plot(knn_or[0],knn_or[1], 'b', label='Original_Data')
line5=plt.plot(knn_ov[0],knn_ov[1], 'r' , label='OverSampling')
line6=plt.plot(knn_un[0],knn_un[1], 'g' , label='UnderSampling')
plt.legend([line4])
plt.legend([line5])
plt.legend([line6])
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('KNN')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

#ROC_XG
xg_or = []
xg_ov = []
xg_un = []
xg_or.append(FPR[2])
xg_ov.append(FPR_over[2])
xg_un.append(FPR_under[2])
xg_or.append(TPR[2])
xg_ov.append(TPR_over[2])
xg_un.append(TPR_under[2])

line7=plt.plot(xg_or[0],xg_or[1], 'b', label='Original_Data')
line8=plt.plot(xg_ov[0],xg_ov[1], 'r' , label='OverSampling')
line9=plt.plot(xg_un[0],xg_un[1], 'g' , label='UnderSampling')
plt.legend([line7])
plt.legend([line8])
plt.legend([line9])
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('XGBoost')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

#######Tuning Parameters#######
from sklearn.model_selection import GridSearchCV
def Classification_model_GridSearchCV(model,Data,x,y,params):
  data_x = Data.ix[:,x]
  data_y = Data.ix[:,y]
  data_y = data_y.values.ravel()
  clf = GridSearchCV(model, params,scoring='accuracy',cv=5)
  clf.fit(data_x,data_y)
  print('best score is:')
  print(clf.best_score_)
  print('best estimator is:')
  print(clf.best_estimator_)
  return clf.best_score_


Model_Accuracy_Tuning=[]
model=knn()
param_grid={'n_neighbors':[55,15],'weights':('uniform','distance'),'p':[1,5]}
Accuracy=Classification_model_GridSearchCV(model,train_data,All_features,Pred_var,param_grid)
Model_Accuracy_Tuning.append(Accuracy)

model=LogisticRegression()
param_grid={'C': [0.01,0.1,1,10],'penalty':('l1','l2')}
Accuracy=Classification_model_GridSearchCV(model,train_data,All_features,Pred_var,param_grid)
Model_Accuracy_Tuning.append(Accuracy)

Accuracy_with_GridSearchCV = pd.DataFrame(
    { "Classification Model" :models,
     "Accuracy with GridSearchCV":Model_Accuracy_Tuning
     
    })
Accuracy_with_GridSearchCV.sort_values(by="Accuracy with GridSearchCV",ascending=False).reset_index(drop=True)
#######Get coefficients#######
train_x = train_data.ix[:,All_features]
train_y = train_data.ix[:,Pred_var]
model = LogisticRegression().fit(train_x,train_y.values.ravel())
model.coef_


