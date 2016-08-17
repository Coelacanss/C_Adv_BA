# First, some simple experiments with svm
library(e1071)
load(file.choose())  # load in df1 from L06/06 svm.Rda

library(ggplot2)
#plot the points
ggplot(df1, aes(x1,x2)) + 
      geom_point(aes(color=y), shape=20, size=3) 

#fit the hyperplane
# try a very large cost factor, this is like a hard margin, separating hyperplan
fit.df1 <- svm(y~., data=df1, kernel='linear', cost=100, scale=F)  ### General scale = T
summary(fit.df1)
plot(fit.df1, df1)  ### Xs are Support Vectors

# line equations
# hyperplane:  coef1 x1 + coef2 x2 = rho
# margins:     coef1 x1 + coef2 x2 = rho +/- 1
# if scale=T then use fit.df1$SV instead of df1
# fit.df1$index is the set of support vectors   ### in this case, number 3,4,16 are SVs

coef1 = sum(fit.df1$coefs*df1[fit.df1$index,1])
coef2 = sum(fit.df1$coefs*df1[fit.df1$index,2])
sv = c(rep(0,20))
sv[fit.df1$index] = 1
sv = as.factor(sv)
ggplot(df1, aes(x1,x2)) + geom_point(aes(color=y, shape=sv), size=3) +
  geom_abline(intercept = fit.df1$rho/coef2, slope= -coef1/coef2) +
  geom_abline(intercept = (fit.df1$rho+1)/coef2, slope= -coef1/coef2, lty=2) +
  geom_abline(intercept = (fit.df1$rho-1)/coef2, slope= -coef1/coef2, lty=2)

# decreasing the cost factor
fit.df1 <- svm(y~., data=df1, kernel='linear', cost=1, scale=F)
coef1 = sum(fit.df1$coefs*df1[fit.df1$index,1])
coef2 = sum(fit.df1$coefs*df1[fit.df1$index,2])
sv = c(rep(0,20))
sv[fit.df1$index] = 1
sv = as.factor(sv)
ggplot(df1, aes(x1,x2)) + geom_point(aes(color=y, shape=sv), size=3) +
  geom_abline(intercept = fit.df1$rho/coef2, slope= -coef1/coef2) +
  geom_abline(intercept = (fit.df1$rho+1)/coef2, slope= -coef1/coef2, lty=2) +
  geom_abline(intercept = (fit.df1$rho-1)/coef2, slope= -coef1/coef2, lty=2)

# decreasing the cost factor
fit.df1 <- svm(y~., data=df1, kernel='linear', cost=.1, scale=F)
coef1 = sum(fit.df1$coefs*df1[fit.df1$index,1])
coef2 = sum(fit.df1$coefs*df1[fit.df1$index,2])
sv = c(rep(0,20))
sv[fit.df1$index] = 1
sv = as.factor(sv)
ggplot(df1, aes(x1,x2)) + geom_point(aes(color=y, shape=sv), size=3) +
  geom_abline(intercept = fit.df1$rho/coef2, slope= -coef1/coef2) +
  geom_abline(intercept = (fit.df1$rho+1)/coef2, slope= -coef1/coef2, lty=2) +
  geom_abline(intercept = (fit.df1$rho-1)/coef2, slope= -coef1/coef2, lty=2)

# decreasing the cost factor
fit.df1 <- svm(y~., data=df1, kernel='linear', cost=.01, scale=F)
coef1 = sum(fit.df1$coefs*df1[fit.df1$index,1])
coef2 = sum(fit.df1$coefs*df1[fit.df1$index,2])
sv = c(rep(0,20))
sv[fit.df1$index] = 1
sv = as.factor(sv)
ggplot(df1, aes(x1,x2)) + geom_point(aes(color=y, shape=sv), size=3) +
  geom_abline(intercept = fit.df1$rho/coef2, slope= -coef1/coef2) +
  geom_abline(intercept = (fit.df1$rho+1)/coef2, slope= -coef1/coef2, lty=2) +
  geom_abline(intercept = (fit.df1$rho-1)/coef2, slope= -coef1/coef2, lty=2)

# use the tune fuction in svm
set.seed(1)
tune.df1 <- tune(svm, y ~x1+x2, data=df1, kernel='linear',
                 ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
summary(tune.df1)
plot(tune.df1)

# let us try log scale using ggplot
tune.df1$performances   #df of performance
ggplot(tune.df1$performance, aes(x=cost, y=error)) +
  geom_line() + scale_x_log10() +
  theme(text = element_text(size=20))

#
# now look at df2
#
ggplot(df2, aes(x1, x2, color = y)) + geom_point()
set.seed(1)
tune.df2 <- tune(svm, y ~x1+x2, data=df2, kernel='linear',
                 ranges=list(cost=c(0.001,0.01,0.1,1,10,100, 1000)))
ggplot(tune.df2$performance, aes(x=cost, y=error)) +
  geom_line() + scale_x_log10() +
  theme(text = element_text(size=20))
# pretty bad!
# need a different kernel - radial
fit.df2 <- svm(y~x1+x2, data=df2, kernel='radial', cost=.1, scale=F)
plot(fit.df2, df2)

fit.df2 <- svm(y~x1+x2, data=df2, kernel='radial', cost=1, scale=F)
plot(fit.df2, df2)

fit.df2 <- svm(y~x1+x2, data=df2, kernel='radial', cost=10, scale=F)
plot(fit.df2, df2)

fit.df2 <- svm(y~x1+x2, data=df2, kernel='radial', cost=100, scale=F)
plot(fit.df2, df2)

fit.df2 <- svm(y~x1+x2, data=df2, kernel='radial', cost=1000, scale=F)
plot(fit.df2, df2)

fit.df2 <- svm(y~x1+x2, data=df2, kernel='radial', cost=10000, scale=F)
plot(fit.df2, df2)

set.seed(1)
tune.df2 <- tune(svm, y ~x1+x2, data=df2, kernel='radial',
                 ranges=list(cost=c(0.001,0.01,0.1,1,10,100, 1000)))
ggplot(tune.df2$performance, aes(x=cost, y=error)) +
  geom_line() + scale_x_log10() +
  theme(text = element_text(size=20))

summary(tune.df2)
#
# ==== now experiment with default data
#
load(file.choose())  # load in L05/default.rda  

#create test and train samples
set.seed(100)

train = sample(1:nrow(Default),nrow(Default)*0.667)
Default.train = Default[train,]
Default.test = Default[-train,]

# let us visualize the data
ggplot(Default.train, aes(balance, income)) +
  geom_point(aes(color=default, shape=student))

# first with default options kernel: radial 
fit.def = svm(default ~ student + balance + income, 
              data=Default.train)
summary(fit.def)
plot(fit.def, Default.train, balance~income)   #need formula since more than 2 predictors

# now linear kernel
fit.def = svm(default ~ student + balance + income, 
              data=Default.train,
              kernel='linear')
summary(fit.def)
plot(fit.def, Default.train, balance~income)   #need formula since more than 2 predictors

# now polynomial kernel
fit.def = svm(default ~ student + balance + income, 
              data=Default.train,
              kernel='polynomial')
summary(fit.def)
plot(fit.def, Default.train, balance~income)   

#tuning the polynomial kernel
set.seed(1)
tune.def <- tune(svm, default ~ student + balance + income, 
                 data=Default.train,
                 kernel='polynomial',
                 ranges=list(cost=c(0.001,0.01,0.1,1,10,100, 1000)))
ggplot(tune.def$performance, aes(x=cost, y=error)) +
  geom_line() + scale_x_log10() +
  theme(text = element_text(size=20))
# cost = 10 seems to be effective

fit.def = svm(default ~ student + balance + income, 
              data=Default.train,
              kernel='polynomial', cost=10,
              degree=3, coef0=50, gamma=1/2)   
summary(fit.def)
plot(fit.def, Default.train, balance~income)   

# class weights
# for unequal classes, weights can be used to weigh the errors in an unequal fashion
# we are ok with more error on the no class to reduce the error on the yes class
#tuning the polynomial kernel with class weights
set.seed(1)
tune.def <- tune(svm, default ~ student + balance + income, 
                 data=Default.train,
                 kernel='polynomial',
                 degree=3, coef0=50, gamma=1/2,
#                 class.weights=c(No=0.1, Yes=1),
                 ranges=list(cost=c(0.1,1,10,100),
                             class.weights=list(c(No=0.1, Yes=1),
                                                c(No=0.2, Yes=1))))
ggplot(tune.def$performance, aes(x=cost, y=error)) +
  geom_line() + scale_x_log10() +
  theme(text = element_text(size=20))

# now let us check this one out in detail
fit.def = svm(default ~ student + balance + income, 
              data=Default.train,
              kernel='polynomial', cost=1,
              degree=3, coef0=50, gamma=1/2,
              class.weights=c(No=0.1, Yes=1))   
summary(fit.def)
plot(fit.def, Default.train, balance~income)   

# now for profit evaluation
#
# profit function
#
# lend $100 to every one predicted negative
# get back $110 from true negative, get back $50 from false negative
# profit = 10* TN - 50* FN
profit = function(cm) { #cm confusion matrix No class 1, yes clas 2, rows are predicted, cols are actual
  return(10*cm[1,1]-50*cm[1,2])
}

# let us set up the cm matrix if we use the null classifier of
#  classifying all observations as non-default
summary(Default.test)
# test has 3,233 actual no and 97 actual yes
cm = matrix(c(3233,97,0,0),
            nrow=2,              # number of rows 
            ncol=2,              # number of columns 
            byrow = TRUE)        # matrix(c(), nrow=2)
profit.null = profit(cm)
profit.null   #  $27,480
# tree 29020 - 27480 = $1,540
# linear regression  29420 - 27480 = $1,940
# logistic regression 29530 - 27480 = $2,050

# for SVM:
default.pred = predict(fit.def, Default.test[,-1])
cm = table(default.pred, Default.test$default)
cm
profit(cm)    #  $29,460  
profit(cm) - profit.null  # $1,980

