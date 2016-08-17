# load in default.Rda
str(Default)
head(Default)
summary(Default)   #extremely unbalanced, only 3.33% defaults

#create a numerical version of default for numeric methods
# ndef = 1 if default, 0 other wise
# nstud = 1 if student
Default$ndef = ifelse(Default$default=='Yes',1,0)
Default$nstud = ifelse(Default$student=='Yes',1,0)

# let us split it into training and testing datasets
set.seed(100)
train = sample(1:nrow(Default),nrow(Default)*0.667)
Default.train = Default[train,]
Default.test = Default[-train,]

# let us explore the training data
summary(Default.train[,1:4])

prop.table(xtabs(~ student + default, data=Default.train),1) # 1 row percentages, 2 col
aggregate(cbind(balance, income) ~ default, Default.train, mean)

# graphical exploration of training data

library(ggplot2)
ggplot(Default.train, aes(balance, income, color=default)) + 
  geom_point() + theme(text = element_text(size=20))

ggplot(Default.train, aes(default, balance, color=default, fill=default)) + 
  geom_boxplot(alpha=1/5)+ theme(text = element_text(size=20))

ggplot(Default.train, aes(default, income, color=default, fill=default)) + 
  geom_boxplot(alpha=1/5)+ theme(text = element_text(size=20))

ggplot(Default.train, aes(default, balance, color=default)) + 
  geom_violin(aes(fill=default), alpha=1/3) +
  theme(text = element_text(size=20)) 

ggplot(Default.train, aes(default, income, color=default)) + 
  geom_violin(aes(fill=default), alpha=1/3)+
  theme(text = element_text(size=20))

#
# profit function
#
# lend $100 to every one predicted negative
# get back $110 from true negative, get back $50 from false negative
# profit = 10* TN - 50* FN
profit = function(cm) { #cm confusion matrix No class 1, yes clas 2, rows are predicted, cols are actual
  return(10*cm[1,1]-50*cm[1,2])
}

#
# TREE
#
library(rpart)

fit = rpart(default ~ balance+income+student, 
            data=Default.train, method="class",
            control=rpart.control(xval=10, minsplit=100))
nrow(fit$frame)  # 3 nodes, 1 split

#plot the  tree
plot(fit, uniform=T, branch=0, compress=T,
     main="Tree for Default", margin=0.1)
text(fit,  splits=T, all=F, use.n=T, 
     pretty=T, fancy=F, cex=1.2)

# let us look at the probabilistic predictions
default.prob = predict(fit, Default.train, type="prob")[,2]
ggplot(as.data.frame(default.prob), aes(default.prob)) + geom_histogram(binwidth=0.01)
# it is bimodal since we only have two leaf nodes
# So threshold default of 0.5 is just fine

# the confusion matrix
default.pred = predict(fit, Default.test, type="class")
cm = table(default.pred, Default.test$default)
cm

profit(cm)
# best profit $29,020

#
# LINEAR REGRESSION
#
lm.fit = lm(ndef ~ balance+income+nstud, data=Default.train)
lm.fit
summary(lm.fit)

# do the threshold tuning on training data
default.prob = predict(lm.fit, Default.train)
default.actual=Default.train$default
# let us take a look at the histogram of probability predictions
ggplot(data.frame(default.prob, default.actual), aes(default.prob)) + 
  geom_histogram(aes(fill=default.actual),binwidth=0.01, color='white') +
  theme(text=element_text(size=20)) +
  labs(x='Predicted Probability', y='Frequency')

threshold.values = seq(0.1, 0.2, 0.005)
profit.values = c()
for (i in 1:length(threshold.values)) {
  default.pred = ifelse(default.prob < threshold.values[i], 0, 1)
  profit.values[i] = profit(table(default.pred, Default.train$ndef))
}

#now let us plot the profit vs threshold
plot(threshold.values,profit.values)
lines(threshold.values,profit.values) 
#pick the best based on training data
opt.threshold = threshold.values[which.max(profit.values)]  
opt.threshold   # 0.145 is the profit maximizing threshold

# now the threshold in terms of balance
bal = (opt.threshold - lm.fit$coefficients[1])/lm.fit$coefficients[2]
bal  # 1666.72

plot(Default.train$balance, Default.train$ndef, col='red',
     xlab="Balance", ylab="Default: No=0 Yes=1")
abline(lm.fit, lwd=2)
abline(v=1853, lwd=2, lty=2)  # the tree threshold
abline(v=bal, lwd=2, lty=1)  # the tree threshold

#now let us evaluate the tuned model on test data
cm = table(ifelse(predict(lm.fit, Default.test) < opt.threshold, 0, 1), Default.test$ndef)
cm
profit(cm)
# this gives $29,490


#
# LOGISTIC REGRESSION
#

glm.fit = glm(ndef ~ balance+income+nstud, data=Default.train, family = binomial)
summary(glm.fit)

# notice that both balance and nstud are significant

# must have type="response" or you will get logit predictions
default.prob = predict(glm.fit, Default.train, type="response") 
default.actual=Default.train$default
# let us take a look at the histogram of probability predictions
# define a new column in Default called status
Default.train$status = 'NN'
Default.train$status[Default.train$default=='Yes'&Default.train$student=='Yes'] = 'DS'
Default.train$status[Default.train$default=='Yes'&Default.train$student=='No'] = 'DN'
Default.train$status[Default.train$default=='No'&Default.train$student=='Yes'] = 'NS'
Default.train$status = as.factor(Default$status)
summary(Default.train$status)
ggplot(data.frame(default.prob, status=Default.train$status), aes(default.prob)) + 
  geom_histogram(aes(fill=status),binwidth=0.05) +
  theme(text=element_text(size=20)) +
  labs(x='Predicted Probability', y='Frequency')

#detail of the tail
ggplot(data.frame(default.prob, status=Default.train$status), aes(default.prob)) + 
  geom_histogram(aes(fill=status),binwidth=0.05) +
  theme(text=element_text(size=20)) +
  labs(x='Predicted Probability', y='Frequency') +
  xlim(0.05, 1)

# Tune the threshold
threshold.values = seq(0.0, 0.50, 0.05)
profit.values = c()
for (i in 1:length(threshold.values)) {
  default.pred = ifelse(default.prob < threshold.values[i],'No','Yes')
  profit.values[i] = 
    profit(table(default.pred, default.actual))
}

plot(threshold.values, profit.values)
lines(threshold.values, profit.values) 
opt.threshold = threshold.values[which.max(profit.values)]  
opt.threshold   # 0.2 is the optimal threshold

# now make the predictions for the test set and predict the profit from it
default.prob.test = predict(glm.fit, Default.test, type="response") 
default.pred.test = ifelse(default.prob.test < opt.threshold, 'No', 'Yes')
cm = table(default.pred.test, Default.test$default)
cm
profit(cm)
# profit 29,530

# Now let us try and visualize the difference
# now compute the bal threshold for each case
# students
y.threshold = log(opt.threshold/(1+opt.threshold))
bal.threshold.1 = 
  (y.threshold - glm.fit$coefficients[1]- glm.fit$coefficients[4])/glm.fit$coefficients[2]
bal.threshold.1
# balance threshold for students $1676

# non-students
bal.threshold.0 = 
  (y.threshold - glm.fit$coefficients[1])/glm.fit$coefficients[2]
bal.threshold.0
# balance threshold for non-students $1568

ggplot(Default.train, aes(x=balance, y=ndef)) + 
  geom_point(aes(color=student), solid=F, shape=1, size=4 ) + 
  stat_smooth(method="glm", family="binomial", se=FALSE, color='black', lwd=1) +
  geom_vline(xintercept = bal.threshold.0, color="#F8766D", lwd=1, lty=2)+
  geom_vline(xintercept = bal.threshold.1, color="#00BFC4", lwd=1, lty=2)+
  theme(text = element_text(size=20)) +
  labs(y='Probability of Default', x='Balance')





