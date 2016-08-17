### This script is for Advanced BA course Assignment #4, form Team 3
library(caret)
library(plyr)
library(e1071)

## load data
data.raw <- read.csv(file = 'churn.csv')
summary(data.raw)
nearZeroVar(data.raw, saveMetrics = TRUE)

## 1.split data into training set and testing set
set.seed(100)
train <- sample(1:nrow(data.raw), nrow(data.raw) * 0.667)
data.train = data.raw[train, ]
data.test = data.raw[-train, ]

## 2.Visualize the key features and class variable using the training subset.
# check distribution of class variable
ggplot(data = data.train) +
      geom_bar(aes(stay), width = 0.3, fill = 'darkred')
# check the relation between class variable and key features
# class variable vs. college
ggplot(data = data.train) +
      geom_bar(aes(stay, fill = college), width = 0.3, position = 'dodge')
# class variable vs. satisfaction
ggplot(data = data.train) +
      geom_bar(aes(stay, fill = rep_sat), width = 0.3, position = 'dodge')
# class variable vs. usage
ggplot(data = data.train) +
      geom_bar(aes(stay, fill = rep_usage), width = 0.3, position = 'dodge')
# class variable vs. consideration of changing plans
ggplot(data = data.train) +
      geom_bar(aes(stay, fill = rep_change), width = 0.3, position = 'dodge')

## The svm() library requires the class variable to be a factor and all other features to be numeric.
## 4.Convert the college feature, a two-level factor to a binary variable.
college = as.numeric(data.raw$college)
college[college == 2] = 0
data.raw$college = college

## 5.Convert the ordinal values of rep_sat, rep_usage and rep_change
# fix rep_sat
count(data.raw$rep_sat)
data.raw$rep_sat = factor(data.raw$rep_sat, levels(data.raw$rep_sat)[c(5,3,1,2,4)])
data.raw$rep_sat = as.numeric(data.raw$rep_sat)
count(data.raw$rep_sat)
# fix rep_usage
count(data.raw$rep_usage)
data.raw$rep_usage = factor(data.raw$rep_usage, levels(data.raw$rep_usage)[c(5,3,1,2,4)])
data.raw$rep_usage = as.numeric(data.raw$rep_usage)
count(data.raw$rep_usage)
# fix rep_change
count(data.raw$rep_change)
data.raw$rep_change = factor(data.raw$rep_change, levels(data.raw$rep_change)[c(3,4,5,2,1)])
data.raw$rep_change = as.numeric(data.raw$rep_change)
count(data.raw$rep_change)

## 6. Split the cleaned up dataframe again into test and training samples
data.train = data.raw[train, ]
data.test = data.raw[-train, ]

## 7. Picking kernal and tuning parameters using 10% sample of training data
# create sample
set.seed(100)
sample <- sample(1:nrow(data.train), nrow(data.train) * 0.1)
data.train.sample = data.train[sample, ]

## 8. picking and tuning
# a. try linear
set.seed(100)
tune.linear <- tune(svm, stay ~., data=data.train.sample, kernel='linear',
                 ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
summary(tune.linear)
ggplot(tune.linear$performance, aes(x=cost, y=error)) +
      geom_line() + scale_x_log10() +
      theme(text = element_text(size=20))
# narrow
set.seed(100)
tune.linear <- tune(svm, stay ~., data=data.train.sample, kernel='linear',
                    ranges=list(cost=c(seq(0.1,2,by = 0.1))))
summary(tune.linear)
ggplot(tune.linear$performance, aes(x=cost, y=error)) +
      geom_line() +
      theme(text = element_text(size=20))
# narrow
set.seed(100)
tune.linear <- tune(svm, stay ~., data=data.train.sample, kernel='linear',
                    ranges=list(cost=c(seq(1.4,1.6,by = 0.01))))
summary(tune.linear)
ggplot(tune.linear$performance, aes(x=cost, y=error)) +
      geom_line() +
      theme(text = element_text(size=20))
# best cost: 1.5, error: 0.3642408

# b. try radial
set.seed(100)
tune.radial <- tune(svm, stay ~., data=data.train.sample, kernel='radial',
                    ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
summary(tune.radial)
ggplot(tune.radial$performance, aes(x=cost, y=error)) +
      geom_line() + scale_x_log10() +
      theme(text = element_text(size=20))
# narrow
set.seed(100)
tune.radial <- tune(svm, stay ~., data=data.train.sample, kernel='radial',
                    ranges=list(cost=c(seq(0.1,2,by = 0.1))))
summary(tune.radial)
ggplot(tune.radial$performance, aes(x=cost, y=error)) +
      geom_line() +
      theme(text = element_text(size=20))
# narrow
set.seed(100)
tune.radial <- tune(svm, stay ~., data=data.train.sample, kernel='radial',
                    ranges=list(cost=c(seq(0.6,0.8,by = 0.01))))
summary(tune.radial)
ggplot(tune.radial$performance, aes(x=cost, y=error)) +
      geom_line() +
      theme(text = element_text(size=20))
# best 0.64, error: 0.3432667

# c. try polynomial
set.seed(100)
tune.poly <- tune(svm, stay ~., data=data.train.sample, kernel='polynomial',
                    ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
summary(tune.poly)
ggplot(tune.poly$performance, aes(x=cost, y=error)) +
      geom_line() + scale_x_log10() +
      theme(text = element_text(size=20))
# narrow
set.seed(100)
tune.poly <- tune(svm, stay ~., data=data.train.sample, kernel='polynomial',
                  ranges=list(cost=c(seq(0.05,1,by = 0.05))))
summary(tune.poly)
ggplot(tune.poly$performance, aes(x=cost, y=error)) +
      geom_line() +
      theme(text = element_text(size=20))
# narrow
set.seed(100)
tune.poly <- tune(svm, stay ~., data=data.train.sample, kernel='polynomial',
                  ranges=list(cost=c(seq(0.1,0.25,by = 0.01))))
summary(tune.poly)
ggplot(tune.poly$performance, aes(x=cost, y=error)) +
      geom_line() +
      theme(text = element_text(size=20))
# best cost:0.20  error: 0.3687128

## So we pick 'radial' kernal and cost = 0.64 for modeling
## 9.Build model
time.now = proc.time()
fit = svm(stay ~., data=data.train, kernel='radial', cost=0.64) 
proc.time() - time.now

## 10.make predictions on the test data set, compute the confusion matrix and analyze
pred = predict(fit, data.test)
confusionMatrix(pred, reference = data.test$stay)
# accuracy:0.6739

## 11.build a tree model for comparison
library(rpart)
time.now = proc.time()
fit.tree = rpart(stay ~ ., data=data.train, method="class", 
                 control=rpart.control(xval=10, minsplit=10))
proc.time() - time.now

pred.tree = predict(fit.tree, data.test, type = 'class')
confusionMatrix(pred.tree, reference = data.test$stay)
# accuracy:0.6865

## So decision tree gives better accuracy

