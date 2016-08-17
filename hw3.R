### This script is for Advanced BA course assignment #3, from Team 3
library(ggplot2)

## 1.Load data and check structure
data.raw <- read.csv(file = 'ccFraud.csv')
head(data.raw) 
sapply(data.raw, function(x) sum(is.na(x)))  # no missing value
# drop custID since it does not influence the accessment result
data.raw$custID = NULL
# drop state. Though we could use one-hot encoding to convert, it would take much longer time to
# train model if we get 50 more features.
data.raw$state = NULL
# convert gender and cardholder into 0/1 variables
data.raw$gender = data.raw$gender - 1
data.raw$cardholder = data.raw$cardholder - 1

## 2.Partitition the data into 80% training, 20% test using set.seed(100)
set.seed(100)
train = sample(1:nrow(data.raw),nrow(data.raw)*0.8)
data.train = data.raw[train,]
data.test = data.raw[-train,]

## 3.Visualize the problem and training data
# exploration
summary(data.train[,1:7])

ggplot(data.train, aes(balance, numTrans, color=fraudRisk)) + 
      geom_point() + 
      theme(text = element_text(size=20))

ggplot(data.train, aes(fraudRisk, balance, color=fraudRisk, fill=fraudRisk)) + 
      geom_boxplot(alpha=1/5) + 
      theme(text = element_text(size=20))

## 4.Build a variety of models (tree, linear regression, logistic regression) and
## tune them using training data, and then find the maximum profit using test data.

# create a function for calculating profit
profit = function(cm) {
      p = 2*cm[1,1]-100*cm[1,2]-10*cm[2,1]
      return(p)
}

###################### TREE MODEL ####################
#
library(rpart)
time.now = proc.time()
fit.tree = rpart(fraudRisk ~ ., data=data.train, method="class", 
                 control=rpart.control(xval=5, minsplit=100))
proc.time() - time.now
save(fit.tree, file = 'model.tree.val5.Rds')

# hist of probabilistic predictions
fraud.prob = predict(fit.tree, data.train, type="prob")[,2]
ggplot(as.data.frame(fraud.prob), aes(fraud.prob)) + 
      geom_histogram(binwidth=0.01) +
      labs(x='Predicted Probability', y='Frequency')

# tune threshhold
table(fraud.prob)
threshold.values = c(0.1, 0.245, 0.45, 0.655, 0.694, 0.71)
profit.values = c()
for (i in 1:length(threshold.values)) {
      fraud.pred = ifelse(fraud.prob < threshold.values[i], 0, 1)
      profit.values[i] = profit(table(fraud.pred, data.train$fraudRisk))
}

# plot the profit vs threshold
plot(threshold.values,profit.values)
lines(threshold.values,profit.values) 
# pick the best based on training data
opt.threshold = threshold.values[which.max(profit.values)]  
opt.threshold   # 0.1

# show confusion matrix and profit
time.now = proc.time()
pred.tree = predict(fit.tree, data.test, type = 'prob')[,2]
pred.tree = ifelse(pred.tree < opt.threshold, 0, 1)
time.tree = proc.time() - time.now
time.tree  # 2.586s

cm = table(pred.tree, data.test$fraudRisk)
cm
profit(cm)  # maximum profit $-2,613,686
TN.tree = 2*cm[1,1]
FN.tree = -100*cm[1,2]
FP.tree = -10*cm[2,1]

################# LINEAR REGRESSION ###################
#
fit.lm = lm(fraudRisk ~ ., data = data.train)
summary(fit.lm)

# do the threshold tuning on training data
fraud.prob = predict(fit.lm, data.train)
fraud.actual=data.train$fraudRisk
# histogram of probability predictions
ggplot(as.data.frame(fraud.prob), aes(fraud.prob, fill = as.factor(fraud.actual))) + 
      geom_histogram(binwidth=0.01) +
      labs(x='Predicted Probability', y='Frequency')

# tune threshold
threshold.values = seq(0.05, 0.6, 0.05)
profit.values = c()
for (i in 1:length(threshold.values)) {
      fraud.pred = ifelse(fraud.prob < threshold.values[i], 0, 1)
      profit.values[i] = profit(table(fraud.pred, data.train$fraudRisk))
}

# plot the profit vs threshold
plot(threshold.values,profit.values)
lines(threshold.values,profit.values)

# narrow the range
threshold.values = seq(0.15, 0.25, 0.005)
profit.values = c()
for (i in 1:length(threshold.values)) {
      fraud.pred = ifelse(fraud.prob < threshold.values[i], 0, 1)
      profit.values[i] = profit(table(fraud.pred, data.train$fraudRisk))
}

# plot the profit vs threshold
plot(threshold.values,profit.values)
lines(threshold.values,profit.values)

# pick the best based on training data
opt.threshold = threshold.values[which.max(profit.values)]  
opt.threshold   # 0.195 is the profit maximizing threshold

# evaluate the tuned model on test data
time.now = proc.time()
pred.lm = predict(fit.lm, data.test)
pred.lm = ifelse(pred.lm < opt.threshold, 0, 1)
time.lm = proc.time() - time.now
time.lm  # 1.401s

cm = table(pred.lm, data.test$fraudRisk)
cm
TN.lm = 2*cm[1,1]
FN.lm = -100*cm[1,2]
FP.lm = -10*cm[2,1]
profit(cm)
# this gives $-239,162 with state / -250,902 without state


################# LOGISTIC REGRESSION ###################
#
fit.glm = glm(fraudRisk ~ ., data = data.train, family = binomial)
summary(fit.glm)

# do the threshold tuning on training data
fraud.prob = predict(fit.glm, data.train, type='response', family = binomial)
fraud.actual = data.train$fraudRisk
# histogram of probability predictions
ggplot(as.data.frame(fraud.prob), aes(fraud.prob, fill = as.factor(fraud.actual))) + 
      geom_histogram(binwidth=0.01) +
      labs(x='Predicted Probability', y='Frequency')

# tune threshold
threshold.values = seq(0.05, 0.6, 0.05)
profit.values = c()
for (i in 1:length(threshold.values)) {
      fraud.pred = ifelse(fraud.prob < threshold.values[i], 0, 1)
      profit.values[i] = profit(table(fraud.pred, data.train$fraudRisk))
}

# plot the profit vs threshold
plot(threshold.values,profit.values)
lines(threshold.values,profit.values)

# narrow the range
threshold.values = seq(0.15, 0.25, 0.005)
profit.values = c()
for (i in 1:length(threshold.values)) {
      fraud.pred = ifelse(fraud.prob < threshold.values[i], 0, 1)
      profit.values[i] = profit(table(fraud.pred, data.train$fraudRisk))
}

# pick the best based on training data
opt.threshold = threshold.values[which.max(profit.values)]  
opt.threshold  # 0.195 is the profit maximizing threshold

# evaluate the tuned model on test data
time.now = proc.time()
pred.glm = predict(fit.glm, data.test)
pred.glm = ifelse(pred.glm < opt.threshold, 0, 1)
time.glm = proc.time() - time.now
time.glm  # 1.387s

cm = table(pred.glm, data.test$fraudRisk)
cm
TN.glm = 2*cm[1,1]
FN.glm = -100*cm[1,2]
FP.glm = -10*cm[2,1]
profit(cm)
# this gives $-239,162 with state / -250,902 without state


###########  REPORT ############
#
model_name = c('tree', 'linear regression', 'logistic regression')
opt_threshold = c(0.1, 0.195, 0.195)
time = c(time.tree[3], time.lm[3], time.glm[3])
max_profit = c(-2613686, -250902, -250902)
TN_value = c(TN.tree, TN.lm, TN.glm)
FN_cost = c(FN.tree, FN.lm, FN.glm)
FP_cost = c(FP.tree, FP.lm, FP.glm)
data.result <- data.frame(model_name, opt_threshold, time, max_profit, TN_value, FN_cost, FP_cost)

ggplot(data = data.result) + 
      theme_bw() +
      geom_bar(aes(model_name, max_profit, fill = model_name), stat = 'identity') +
      theme(axis.text.x = element_text(angle = 30, hjust = 1)) + 
      ggtitle('Maximum Profit')
dev.copy(jpeg, file="Maximum Profit.jpg")
dev.off()

ggplot(data = data.result) + 
      theme_bw() +
      geom_bar(aes(model_name, time, fill = model_name), stat = 'identity') +
      theme(axis.text.x = element_text(angle = 30, hjust = 1)) + 
      ggtitle('Time Usage')
dev.copy(jpeg, file="Time Usage.jpg")
dev.off()
