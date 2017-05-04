install.packages('neuralnet')
library("neuralnet")
install.packages("caret")
library("caret")
mydata=read.csv("sampleOver2.csv")
features=names(testdata[2:20])
form=paste(features,collapse='+')
form=paste('y~',form)
form=as.formula(form)
newdata=model.matrix(form,mydata)
newdata=newdata[1:nrow(mydata),2:48]
newdata=as.data.frame(newdata)
newdata=cbind(mydata[1],newdata)

features=names(newdata[,2:48])
form=paste(features,collapse='+')
form=paste('y~',form)
form=as.formula(form)
mynn=neuralnet(form,newdata,hidden=5,stepmax=700000,rep=2)

testdata=read.csv("/Users/Dani/Documents/ISYE 6740/Project/Bank Marketing/bank-full-test.csv")
testfeatures=names(testdata[2:20])
testform=paste(testfeatures,collapse='+')
testform=paste('y~',testform)
testform=as.formula(testform)
newtestdata=model.matrix(testform,testdata)
newtestdata=newtestdata[1:3824,2:48]
newtestdata=as.data.frame(newtestdata)
newtestdata=cbind(testdata[1],newtestdata)


#compute for test data
nnpredictions=compute(mynn,newtestdata[,2:48])
rawpredicts=nnpredictions$net.result

#Confusion matrix
table(newtestdata[[1]],rawpredicts)

#plot
plot(mynn)


#ROC/AUC
install.packages("pROC")
library(pROC)
roc(newtestdata[[1]],rawpredicts,auc=TRUE,plot=TRUE)
