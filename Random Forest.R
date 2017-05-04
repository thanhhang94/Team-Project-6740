# dataOrig <- read.table(file = "bank-additional-full-noyear.csv", sep=",", header = TRUE)
# nData = dim(dataOrig)[1]
# table(dataOrig$default)
# 
# data = dataOrig
# month = "may"
# year = matrix(0,nData,1)
# yearNo = 2008
# colnames(year) <- 'year'
# count08 = 1
# while(data[count08,10] != 'mar'){
#   year[count08,] = yearNo
#   count08 = count08 +1
# }
# yearNo = yearNo + 1
# while(data[count08,10] == 'mar'){
#   year[count08,] = yearNo
#   count08 = count08 +1
# }
# while(data[count08,10] != 'mar'){
#   year[count08,] = yearNo
#   count08 = count08 +1
# }
# yearNo = yearNo + 1
# while(count08 <= nData){
#   year[count08,] = yearNo
#   count08 = count08 +1
# }
# dataYear = cbind(data,year)
# levels(dataYear$y) <- c('0','1')
# table(dataYear$y)
# write.table(dataYear, file ="bank-addition-full-year.csv", sep=
#                              ',', row.names = FALSE, col.names = TRUE)



# data <- read.table(file = "bank-addition-full-year.csv", sep=",", header = TRUE)
# set.seed(2017)
# n = dim(data)[1]
# n1 = round(n/10)
# flag <- sort(sample(1:n,n1))
# dataTrain <- data[-flag,]
# dataTest <- data[flag,]
# write.table(flag, file ="bank-flag.csv", sep=
#               ',', row.names = FALSE, col.names = TRUE)
# write.table(dataTrain, file ="bank-full-train.csv", sep=
#               ',', row.names = FALSE, col.names = TRUE)
# write.table(dataTest, file ="bank-full-test.csv", sep=
#               ',', row.names = FALSE, col.names = TRUE)
# 
# 
# dataYearO <- read.table(file = "bank-addition-full-year.csv", sep=",", header = TRUE)
# dataYear <- dataYearO
# dataYear$job = as.numeric(dataYear$job)
# dataYear$marital = as.numeric(dataYear$marital)
# dataYear$education = as.numeric(dataYear$education)
# dataYear$default = as.numeric(dataYear$default)
# dataYear$housing = as.numeric(dataYear$housing)
# dataYear$loan = as.numeric(dataYear$loan)
# dataYear$contact = as.numeric(dataYear$contact)
# dataYear$month = as.numeric(dataYear$month)
# dataYear$day_of_week = as.numeric(dataYear$day_of_week)
# dataYear$poutcome = as.numeric(dataYear$poutcome)
# test_smote = dataYear[flag,]
# train_smote = dataYear[-flag,]
# write.table(train_smote, file ="bank-full-allnum-train.csv", sep=
#            ',', row.names = FALSE, col.names = TRUE)
# write.table(test_smote, file ="sampleSmoteTest.csv", sep=
#               ',', row.names = FALSE, col.names = TRUE)
# 

#### import training data
data0 <- read.table(file = "bank-full-train.csv", sep=",", header = TRUE)
dataUnder <- read.table(file = "sampleUnder2.csv", sep=",", header = TRUE)
dataOver <- read.table(file = "sampleOver2.csv", sep=",", header = TRUE)
dataSmote <-read.table(file = "sampleSmote2.csv", sep=",", header = TRUE)
data0$y = as.factor(data0$y)
dataUnder$y = as.factor(dataUnder$y)
dataOver$y = as.factor(dataOver$y)
dataSmote$y = as.factor(dataSmote$y)
#### import testing data
dataTest = read.table(file = "bank-full-test.csv", sep=",", header = TRUE)
dataTestSmote = read.table(file = "sampleSmoteTest.csv", sep=",", header = TRUE)

#### plot the distribution of each dataset
barplotData = data.frame(response_type = c('yes','no','yes','no','yes','no','yes','no'),
                         Dataset = c('Original','Original','UnderSample','UnderSample',
                                     'OverSample','OverSample', 'Smote','Smote'), 
                         percentage = c(3862, 30559, 3800, 5000, 23531, 26469, 30559,30559))
library(ggplot2)
ggplot(barplotData, aes(x=factor(Dataset),y=percentage,fill=factor(response_type))) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_fill_brewer(palette = "Set1")+ggtitle("Distribution")

#### train 4 datasets
library(randomForest)
library(MASS)
data.rf1 <- randomForest(y~., data = data0,
                       importance = TRUE, do.trace=100)
data.rf2 <- randomForest(y~., data = dataUnder,
                         importance = TRUE, do.trace=100)
data.rf3 <- randomForest(y~., data = dataOver,
                         importance = TRUE, do.trace=100)
data.rf4 <- randomForest(y~., data = dataSmote,
                         importance = TRUE, do.trace=100)
#### resample
data.rf5 <- randomForest(y~., data = dataSmote, sampsize = c(5000,3800),
                         importance = TRUE, do.trace=100)


barplotData = data.frame(error_type = c('oob','class1 error','class2 error','oob','class1 error','class2 error',
                                      'oob','class1 error','class2 error','oob','class1 error','class2 error'),
                         Dataset = c('Original','Original','Original', 'UnderSample','UnderSample','UnderSample',
                                     'OverSample','OverSample','OverSample', 'Smote','Smote','Smote'), 
                         percentage = c(8.48, 3.63, 46.84, 11.48, 13.90, 8.29, 3.90, 7.37, 0.00, 4.41, 4.27, 4.55))
library(ggplot2)
ggplot(barplotData, aes(x=factor(Dataset),y=percentage,fill=factor(error_type))) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_fill_brewer(palette = "Set1")+ggtitle("error for 4 different dataset (Random Forest)")

library(caret)
pred10 <- predict(data.rf1, newdata = dataTest[,-1],type="prob")
pred20 <- predict(data.rf2, newdata = dataTest[,-1],type="prob")
pred30 <- predict(data.rf3, newdata = dataTest[,-1],type="prob")
pred40 <- predict(data.rf4, newdata = dataTestSmote[,-1],type="prob")

pred11 <- predict(data.rf1, newdata = dataTest[,-1],type = "response")
confusion11 = confusionMatrix(pred11, dataTest[,1])
pred21 <- predict(data.rf2, newdata = dataTest[,-1],type = "response")
confusion21 = confusionMatrix(pred21, dataTest[,1])
pred31 <- predict(data.rf3, newdata = dataTest[,-1],type = "response")
confusion31 = confusionMatrix(pred31, dataTest[,1])
pred41 <- predict(data.rf4, newdata = dataTestSmote[,-1],type = "response")
confusion41 = confusionMatrix(pred41, dataTestSmote[,1])
cm1 = data.frame(confusion11$table)
cm2 = data.frame(confusion21$table)
cm3 = data.frame(confusion31$table)
cm4 = data.frame(confusion41$table)

ggplot(data = cm1, mapping = aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "skyblue", high = "blue") +
  theme_bw() + theme(legend.position = "none") + ggtitle("original class weight")
ggplot(data = cm2, mapping = aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "skyblue", high = "blue") +
  theme_bw() + theme(legend.position = "none") + ggtitle("Under Sample")
ggplot(data = cm3, mapping = aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "skyblue", high = "blue") +
  theme_bw() + theme(legend.position = "none") + ggtitle("Over Sample")
ggplot(data = cm4, mapping = aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "skyblue", high = "blue") +
  theme_bw() + theme(legend.position = "none") + ggtitle("Smote")

library(ROCR)
#### try2 this one gives reasonable outcome when choose no to plot
pred.obj1 <- prediction(pred10[,2], dataTest[,1])
pred.obj2 <- prediction(pred20[,2], dataTest[,1])
pred.obj3 <- prediction(pred30[,2], dataTest[,1])
pred.obj4 <- prediction(pred40[,2], dataTestSmote[,1])

ROC.perf1 <- performance(pred.obj1, "tpr", "fpr")
ROC.perf2 <- performance(pred.obj2, "tpr", "fpr")
ROC.perf3 <- performance(pred.obj3, "tpr", "fpr")
ROC.perf4 <- performance(pred.obj4, "tpr", "fpr")

AUC.perf1 <- as.numeric(performance(pred.obj1 , "auc")@y.values)
AUC.perf2 <- as.numeric(performance(pred.obj2 , "auc")@y.values)
AUC.perf3 <- as.numeric(performance(pred.obj3 , "auc")@y.values)
AUC.perf4 <- as.numeric(performance(pred.obj4 , "auc")@y.values)

plot(ROC.perf1, main = "ROC plot")
abline(a=0, b=1, lty=2) # diagonal line
plot(ROC.perf2, add=TRUE, col="blue")
plot(ROC.perf3, add=TRUE, col="orange")
plot(ROC.perf4, add=TRUE, col="green")
legend(x=0.67, y=0.46, legend=c("original AUC = 0.9459","underSample AUC = 0.9470",
                               "overSample AUC = 0.9443","Smote AUC = 0.9457"), 
       lty=c(1, 1, 1,1), lwd=c(2, 2, 2,2), cex = 0.5,
       col=c("black", "blue","red", "green"))

barplotData = data.frame(error_type = c('oob','class1 error','class2 error','oob','class1 error','class2 error',
                                        'oob','class1 error','class2 error','oob','class1 error','class2 error'),
                         Dataset = c('Original','Original','Original', 'UnderSample','UnderSample','UnderSample',
                                     'OverSample','OverSample','OverSample', 'Smote','Smote','Smote'), 
                         percentage = c(8.26, 3.44, 50, 12.97, 13.68, 6.82, 9.21, 7.00, 28.28, 8.53, 3.33, 45.96))
library(ggplot2)
ggplot(barplotData, aes(x=factor(Dataset),y=percentage,fill=factor(error_type))) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_fill_brewer(palette = "Set1")+ggtitle("error of testing 4 different dataset (Random Forest)")

#### tuning the parameters
oobError <- rep(0,12)
class1error <- rep(0,12)
class2error <- rep(0,12)
for(i in 1:12){
  data.rf <- randomForest(y~., data = dataUnder, mtry = i, 
                          importance = TRUE, do.trace=100)
  oobError[i] <- data.rf$err.rate[500,1]
  class1error[i] <- data.rf$err.rate[500,2]
  class2error[i] <- data.rf$err.rate[500,3]
}
#### parameter plot
plot(c(1,12),c(0,30),type = "n", main ="oob error with different mtry",
     xlab = "mtry value", ylab = 'percentage(%)',width = 30, height = 40)
lines(oobError*100, col = "red",type = "o")
lines(class1error*100, col = "orange",type = "o")
lines(class2error*100, col = "green",type = "o")
legend(x=10, y=60, legend=c("oobError", "Error1", "Error2"), lty = c(1,1,1),
       col = c("red", "orange","green"), cex =0.5)

data.rf <- randomForest(y~., data = dataUnder, mtry = 6, 
                        importance = TRUE, do.trace=100)
library(caret)
varImp(data.rf)
varImpPlot(data.rf,type=2)

dataUnder1 <- dataUnder[,-c(6,7,8,9,15)]
data.rf5 <- randomForest(y~., data = dataUnder, mtry = 6, 
                         importance = TRUE, do.trace=100)
pred00 <- predict(data.rf, newdata = dataTest[,-1],type="prob")
pred50 <- predict(data.rf5, newdata = dataTest[,-1],type="prob")
pred01 <- predict(data.rf, newdata = dataTest[,-1],type = "response")
pred51 <- predict(data.rf5, newdata = dataTest[,-1],type = "response")
confusion01 = confusionMatrix(pred01, dataTest[,1])
confusion51 = confusionMatrix(pred51, dataTest[,1])
cm0 = confusion01$table
cm5 = confusion51$table

#### finally we choose cm5
pred.obj5 <- prediction(pred50[,2], dataTest[,1])
ROC.perf5 <- performance(pred.obj5, "tpr", "fpr")
AUC.perf5 <- as.numeric(performance(pred.obj5 , "auc")@y.values)
acc = c(confusion51$overall[1])

plot(ROC.perf5, main = "ROC plot for selected model")
abline(a=0, b=1, lty=2) # diagonal line
legend(x=0.82, y=0.46, legend=c("AUC = 0.947","Total Accuracy = 0.870",
                                "Specificity = 0.932","TPR = 0.862"), cex = 0.5, col="black")

##### randomly form new dataUnder
library(randomForest)
library(MASS)
library(ROCR)
library(caret)
data <- read.table(file = "bank-addition-full-year.csv", sep=",", header = TRUE)
data$y = as.factor(data$y)
n = dim(data)[1]
n1 = round(n/10)
flag <- sort(sample(1:n,n1))
dataTrain <- data[-flag,]
dataTest <- data[flag,]
##### split 1/0
dataTrain1 <- subset(dataTrain, y == 1)
n11 = dim(dataTrain1)[1]
flag1 <- sort(sample(1:n11,3800))
dataTrain1 <- dataTrain1[flag1,]

dataTrain0 <- subset(dataTrain, y == 0)
n22 <- dim(dataTrain0)[1]
flag2 <- sort(sample(1:n22,5000))
dataTrain0 <- dataTrain0[flag2,]
newDataTrain <- rbind(dataTrain0,dataTrain1)
data.rf.test <- randomForest(y~., data = newDataTrain, mtry = 6, 
                                        importance = TRUE, do.trace=100)
predTest <- predict(data.rf.test, newdata = dataTest[,-1],type="prob")
predTest <- predict(data.rf.test, newdata = dataTest[,-1],type = "response")
confusionTest = confusionMatrix(predTest, dataTest[,1])
cmTest = confusionTest$table
cmTest
accTest = c(confusionTest$overall[1])
accTest

#### Test by cross validation
library(cvTools)
k <- 5
folds <- cvFolds(NROW(data), K=k)
list.trainoob <- list()
list.testacc <- list()
list.testRecall <- list()

for(i in 1:k){
  dataTrain <- data[folds$subsets[folds$which != i], ] #Set the training set
  dataTest <- data[folds$subsets[folds$which == i], ] #Set the validation set
  
  ##### split 1/0
  n10 <- length(which(dataTrain$y==1))
  dataTrain1 <- subset(dataTrain, y == 1)
  dataTrain0 <- subset(dataTrain, y == 0)
  n22 <- dim(dataTrain0)[1]
  flag2 <- sort(sample(1:n22,5000))
  dataTrain0 <- dataTrain0[flag2,]
  newDataTrain <- rbind(dataTrain0,dataTrain1)
  data.rf.test <- randomForest(y~., data = newDataTrain, mtry = 6, 
                               importance = TRUE, do.trace=100)
  predTest <- predict(data.rf.test, newdata = dataTest[,-1],type="prob")
  predTest <- predict(data.rf.test, newdata = dataTest[,-1],type = "response")
  confusionTest = confusionMatrix(predTest, dataTest[,1])
  list.trainoob[[i]] <- data.rf.test$err.rate[500,1]
  list.testacc[[i]] <- confusionTest$overall[1]
  list.testRecall[[i]] <- confusionTest$table[2,2]/(confusionTest$table[1,2]+confusionTest$table[2,2])
}





data0 <- read.table(file = "bank-full-train.csv", sep=",", header = TRUE)
data0$y = as.factor(data0$y)

dataUnder <- read.table(file = "sampleUnder2.csv", sep=",", header = TRUE)
dataUnder$y = as.factor(dataUnder$y)

dataOver <- read.table(file = "sampleOver2.csv", sep=",", header = TRUE)
dataSmote <-read.table(file = "sampleSmote2.csv", sep=",", header = TRUE)
dataOver$y = as.factor(dataOver$y)
dataSmote$y = as.factor(dataSmote$y)
#### import testing data
dataTest = read.table(file = "bank-full-test.csv", sep=",", header = TRUE)
dataTestSmote = read.table(file = "sampleSmoteTest.csv", sep=",", header = TRUE)

library(randomForest)
library(MASS)
data.rf1 <- randomForest(y~., data = data0,
                         importance = TRUE, do.trace=100)
data.rf2 <- randomForest(y~., data = dataUnder,
                         importance = TRUE, do.trace=100)
data.rf3 <- randomForest(y~., data = dataOver,
                         importance = TRUE, do.trace=100)
data.rf4 <- randomForest(y~., data = dataSmote,
                         importance = TRUE, do.trace=100)

library(randomForest)
library(MASS)
data.rf1 <- randomForest(y~., data = data0,
                         importance = TRUE, do.trace=100)
data.rf2 <- randomForest(y~., data = dataUnder, sampsize = c(5000,3800),
                         importance = TRUE, do.trace=100)
