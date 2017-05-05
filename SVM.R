#data = read.table(file = "/Users/yuewen/Downloads/train.csv", sep = ",", header = TRUE)
#undersampling
data1 = read.table(file = "/Users/yuewen/Downloads/data Elsa/sampleUnder2.csv", sep = ",", header = TRUE)
testdata1 = read.table(file = "/Users/yuewen/Downloads/data Elsa/bank-full-test.csv", sep = ",", header = TRUE)

#smote sampling
data2 = read.table(file = "/Users/yuewen/Downloads/data Elsa/sampleSmoteClean.csv", sep = ",", header = TRUE)
testdata2 = read.table(file = "/Users/yuewen/Downloads/data Elsa/sampleSmoteTest.csv", sep = ",", header = TRUE)

#normal sampling
data3 = read.table(file = "/Users/yuewen/Downloads/data Elsa/bank-full-train.csv", sep = ",", header = TRUE)
testdata3 = read.table(file = "/Users/yuewen/Downloads/data Elsa/bank-full-test.csv", sep = ",", header = TRUE)

#over sampling
data4 = read.table(file = "/Users/yuewen/Downloads/data Elsa/sampleOver2.csv", sep = ",", header = TRUE)
testdata4 = read.table(file = "/Users/yuewen/Downloads/data Elsa/bank-full-test.csv", sep = ",", header = TRUE)


data1$y = factor(data1$y)
data1$job = factor(data1$job)
data1$marital = factor(data1$marital)
data1$education = factor(data1$education)
data1$default = factor(data1$default)
data1$housing = factor(data1$housing)
data1$loan = factor(data1$loan)
data1$contact = factor(data1$contact)
data1$month = factor(data1$month)
data1$day_of_week = factor(data1$day_of_week)
data1$poutcome = factor(data1$poutcome)
data1$year = factor(data1$year)

testdata1$y = factor(testdata1$y)
testdata1$job = factor(testdata1$job)
testdata1$marital = factor(testdata1$marital)
testdata1$education = factor(testdata1$education)
testdata1$default = factor(testdata1$default)
testdata1$housing = factor(testdata1$housing)
testdata1$loan = factor(testdata1$loan)
testdata1$contact = factor(testdata1$contact)
testdata1$month = factor(testdata1$month)
testdata1$day_of_week = factor(testdata1$day_of_week)
testdata1$poutcome = factor(testdata1$poutcome)
testdata1$year = factor(testdata1$year)

data2$y = factor(data2$y)
data2$job = factor(data2$job)
data2$marital = factor(data2$marital)
data2$education = factor(data2$education)
data2$default = factor(data2$default)
data2$housing = factor(data2$housing)
data2$loan = factor(data2$loan)
data2$contact = factor(data2$contact)
data2$month = factor(data2$month)
data2$day_of_week = factor(data2$day_of_week)
data2$poutcome = factor(data2$poutcome)
data2$year = factor(data2$year)
testdata2$y = factor(testdata2$y)
testdata2$job = factor(testdata2$job)
testdata2$marital = factor(testdata2$marital)
testdata2$education = factor(testdata2$education)
testdata2$default = factor(testdata2$default)
testdata2$housing = factor(testdata2$housing)
testdata2$loan = factor(testdata2$loan)
testdata2$contact = factor(testdata2$contact)
testdata2$month = factor(testdata2$month)
testdata2$day_of_week = factor(testdata2$day_of_week)
testdata2$poutcome = factor(testdata2$poutcome)
testdata2$year = factor(testdata2$year)


data3$y = factor(data3$y)
data3$job = factor(data3$job)
data3$marital = factor(data3$marital)
data3$education = factor(data3$education)
data3$default = factor(data3$default)
data3$housing = factor(data3$housing)
data3$loan = factor(data3$loan)
data3$contact = factor(data3$contact)
data3$month = factor(data3$month)
data3$day_of_week = factor(data3$day_of_week)
data3$poutcome = factor(data3$poutcome)
data3$year = factor(data3$year)
testdata3$y = factor(testdata3$y)
testdata3$job = factor(testdata3$job)
testdata3$marital = factor(testdata3$marital)
testdata3$education = factor(testdata3$education)
testdata3$default = factor(testdata3$default)
testdata3$housing = factor(testdata3$housing)
testdata3$loan = factor(testdata3$loan)
testdata3$contact = factor(testdata3$contact)
testdata3$month = factor(testdata3$month)
testdata3$day_of_week = factor(testdata3$day_of_week)
testdata3$poutcome = factor(testdata3$poutcome)
testdata3$year = factor(testdata3$year)


data4$y = factor(data4$y)
data4$job = factor(data4$job)
data4$marital = factor(data4$marital)
data4$education = factor(data4$education)
data4$default = factor(data4$default)
data4$housing = factor(data4$housing)
data4$loan = factor(data4$loan)
data4$contact = factor(data4$contact)
data4$month = factor(data4$month)
data4$day_of_week = factor(data4$day_of_week)
data4$poutcome = factor(data4$poutcome)
data4$year = factor(data4$year)
testdata4$y = factor(testdata4$y)
testdata4$job = factor(testdata4$job)
testdata4$marital = factor(testdata4$marital)
testdata4$education = factor(testdata4$education)
testdata4$default = factor(testdata4$default)
testdata4$housing = factor(testdata4$housing)
testdata4$loan = factor(testdata4$loan)
testdata4$contact = factor(testdata4$contact)
testdata4$month = factor(testdata4$month)
testdata4$day_of_week = factor(testdata4$day_of_week)
testdata4$poutcome = factor(testdata4$poutcome)
testdata4$year = factor(testdata4$year)



### Undersampling 
library(e1071)
svmmodel1 = svm(data1$y~ ., data1)
svmmodel1.predict<-predict(svmmodel1,testdata1[,-1], decision.values = TRUE, type = 'prob')
table(pred = svmmodel1.predict, true = testdata1[,1])


### SMOTE
# data2$y <- as.factor(data2$y)
# svmsmote = SMOTE(y~ ., data2)
# evaluate the SMOTE performance
svmmodel2 <- svm(y ~ ., data = data2)
svmmodel2.predict<-predict(svmmodel2,testdata2[,-1],decision.values=TRUE, type = 'prob')
table(pred = svmmodel2.predict, true = testdata2[,1])

### original
svmmodel3 = svm(data3$y~ ., data3)
svmmodel3.predict<-predict(svmmodel3,testdata3[,-1],decision.values=TRUE, type = 'prob')
table(pred = svmmodel3.predict, true = testdata3[,1])

### oversampling 
svmmodel4 = svm(data4$y~ ., data4)
svmmodel4.predict<-predict(svmmodel4,testdata4[,-1],decision.values=TRUE, type = 'prob')
table(pred = svmmodel4.predict, true = testdata4[,1])

### plot curve
library(gplots)
library(ROCR)
svmmodel1.probs<-attr(svmmodel1.predict,"decision.values")
svmmodel1.labels<-testdata1$y
svmmodel1.prediction<-prediction(svmmodel1.probs,svmmodel1.labels)
svmmodel1.performance<-performance(svmmodel1.prediction,"tpr","fpr")
svmmodel1.auc<-performance(svmmodel1.prediction,"auc")@y.values[[1]]

svmmodel2.probs<-attr(svmmodel2.predict,"decision.values")
svmmodel2.labels<-testdata2$y
svmmodel2.prediction<-prediction(svmmodel2.probs,svmmodel2.labels)
svmmodel2.performance<-performance(svmmodel2.prediction,"tpr","fpr")
svmmodel2.auc<-performance(svmmodel2.prediction,"auc")@y.values[[1]]

svmmodel3.probs<-attr(svmmodel3.predict,"decision.values")
svmmodel3.labels<-testdata3$y
svmmodel3.prediction<-prediction(svmmodel3.probs,svmmodel3.labels)
svmmodel3.performance<-performance(svmmodel3.prediction,"tpr","fpr")
svmmodel3.auc<-performance(svmmodel3.prediction,"auc")@y.values[[1]]

svmmodel4.probs<-attr(svmmodel4.predict,"decision.values")
svmmodel4.labels<-testdata4$y
svmmodel4.prediction<-prediction(svmmodel4.probs,svmmodel4.labels)
svmmodel4.performance<-performance(svmmodel4.prediction,"tpr","fpr")
svmmodel4.auc<-performance(svmmodel4.prediction,"auc")@y.values[[1]]


plot(svmmodel1.performance, main = "ROC plot")
abline(a=0, b=1, lty=2) # diagonal line
plot(svmmodel2.performance, col="blue")
plot(svmmodel3.performance, col="orange")
plot(svmmodel4.performance, col="green")
legend(x=0.67, y=0.46, legend=c("original AUC = 0.9129","underSample AUC = 0.9405",
                                "overSample AUC = 0.9341","Smote AUC = 0.9457"), 
       lty=c(1, 1, 1,1), lwd=c(2, 2, 2,2), cex = 0.5,
       col=c("black", "blue","red", "green"))








### Tunning process
svmmodel<-svm(y~., data=data, method="C-classification",
              kernel="linear", gamma = 0.01, cost = 100,cross=10, probability=TRUE) 
svmmodel
#predicting the test data
svmmodel.predict<-predict(svmmodel,testdata[,2:20],decision.values=TRUE)
svmmodel.probs<-attr(svmmodel.predict,"decision.values")
svmmodel.class<-predict(svmmodel,testdata,type="class")
svmmodel.labels<-testdata$y

#analyzing result
library(lattice)
library(ggplot2)
library(caret)
svmmodel.confusion<-confusion.matrix(svmmodel.labels,svmmodel.class)
svmmodel.accuracy<-prop.correct(svmmodel.confusion)

#roc analysis for test data   ## prediction for the four datasets
library(ROCR)
svmmodel.prediction<-prediction(svmmodel.probs,svmmodel.labels)
svmmodel.performance<-performance(svmmodel.prediction,"tpr","fpr")
svmmodel.auc<-performance(svmmodel.prediction,"auc")@y.values[[1]]



# make the prediction (the dependent variable, Type, has column number 10)

svm.pred <- predict(svm.model, testdata[,2:20])
plot(svm.model, data)
table(pred = svm.pred, true = testdata[,1])

svm.pred <- predict(svm.model, testdata[,2:19], kernel = "radial")

total = rbind(data,testdata)
mytunedsvm <- tune.svm(y ~. , data = total, cost = 10^(-2:-1), epsilon = 10^(1:2)) 
summary(mytunedsvm)
plot (mytunedsvm, transform.x=log10, xlab=expression(log[10](gamma)), ylab="C")


SvmPred = predict(Svm, data[,2:20], probability=TRUE) 
SvmPredRes = table(Pred = SvmPred, True = BankTest[,1]) 




