#
# BIASED SAMPLING AND BOOSTING
#
#
# profit function
#
# spend $100 in marketing to everyone predicted positive
# get back $900 from true positive
# profit = 800* TP - 100* FP
profit = function(cm) { #cm confusion matrix No class 1, yes clas 2, rows are predicted, cols are actual
  return(800*cm[2,2]-100*cm[2,1])
}

# load bank MARKETING data
bm <- read.csv("bank small.csv", sep=";")
bm <- read.csv(file.choose(), sep=";")

summary(bm)

prop.table(table(bm$y))  #only 11% yes

# load the caret library
library(caret)

# multicore processing (MACs only)
library(doMC)
registerDoMC(cores = 2)

#  multicore processing (MAC and PC)
library(doParallel)
# find how many independent threads you can have
makeCluster(detectCores())
# set up to use multiple cores
registerDoParallel(cores=2)

# Now set up training and testing data sets
set.seed(432) # for reproduceable results
# we will use createDataPartition from caret as it creates a sample while preserving
# the class distribution, i.e., it samples within each class
train <- createDataPartition(y=bm$y,  # creates a stratified sample within class
                             p = 0.66667,  # proportion for training
                             list=F)     #put result in a vector
bm.train <- bm[train,]
bm.test <- bm[-train,]

# first cross-validation with a tree model (18 sec)
trc <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5,
                           classProbs = TRUE,
                           allowParallel = T)
set.seed(1)    # for better comparison
tune.cv = train(y ~ ., data=bm.train, 
             method = "rpart", 
             metric = 'Kappa',
             control=rpart.control(minsplit=2, xval=0),
             tuneLength = 20,
             trControl = trc)

tune.cv
plot(tune.cv)
# now for the best tree
fit.tree = tune.cv$finalModel
plot(fit.tree, uniform=TRUE, branch=0.5, 
     main="Classification Tree for Marketing", margin=0.1)
text(fit.tree,  use.n=F, pretty=F, cex=0.6)

nrow(fit.tree$frame)

# now finally check on the test set using test data
pred.cv = predict(tune.cv, bm.test, type='raw')
cm.out = confusionMatrix(pred.cv, bm.test$y, positive='yes')
cm.out
# kappa 0.4272   sesitivity 0.4217  ppr or precision 0.56589
profit(cm.out$table) # $52,800


#
# now let us try weights
#

myweights = ifelse(bm.train$y=='no', 0.2, 1.0)
set.seed(1)    # for better comparison
tune.w = train(y ~ ., data=bm.train, 
                method = "rpart", 
                metric = 'Kappa',
                control=rpart.control(minsplit=2, xval=0),
                tuneLength = 20,
                weights = myweights,
                trControl = trc)
tune.w
plot(tune.w)
# now for the best tree
fit.tree = tune.w$finalModel
plot(fit.tree, uniform=TRUE, branch=0.5, 
     main="Classification Tree for Marketing", margin=0.1)
text(fit.tree,  use.n=F, pretty=F, cex=0.6)

nrow(fit.tree$frame)  # 27

# now finally check on the test set using test data
pred.w = predict(tune.w, bm.test, type='raw')
cm.out = confusionMatrix(pred.w, bm.test$y, positive='yes')
cm.out
# kappa 0.4533   sesitivity 0.6416  ppr or precision 0.4476

profit(cm.out$table) # $80,100

#
# BOOSTING with adabag
#

library(adabag)
set.seed(1)
y.b = boosting(y~., data=bm.train, mfinal=100, boos=T,
                   control=rpart.control(minsplit=2, xval=0),
                   coeflearn = 'Breiman')

summary(y.b)

pred.b = predict.boosting(y.b, bm.test, newfinal=100)
cm.out = confusionMatrix(pred.b$class, bm.test$y, positive='yes')
cm.out
# kappa 0.4388   sesitivity 0.43931  ppr or precision 0.56716
profit(cm.out$table) # $60,600


#
# Creating a balanced sample
# SMOTE  synthetic minority oversampling technique
library(unbalanced)
prop.table(table(bm.train$y))  # 2,667 (88.5%) yes, 348 (11.5%) no
set.seed(223)
bal = ubBalance(bm.train[,-17], bm.train$y, 
                    type='ubSMOTE', positive='yes',  # minority class
                    percOver=200,  # make 2 new samples for every 1 in minority
                    percUnder=200, # how many from majority for each smote
                    k=5,
                    verbose=T)
bal.train = cbind(bal$X, y=bal$Y)
prop.table(table(bal.train$y))  # 1,392 (57%) yes 1,044 (43%) no

detach("package:unbalanced", unload=TRUE)
detach("package:mlr", unload=TRUE)
detach("package:adabag", unload=TRUE)
detach("package:caret", unload=TRUE)

library(caret)
# cross-validation with a tree model on smoted sample
trc <- trainControl(method = "repeatedcv",
                    number = 10,
                    repeats = 5,
                    classProbs = TRUE,
                    allowParallel = T)
set.seed(1)    # for better comparison
tune.bal <- train(y ~ ., data=bal.train, 
                method = "rpart", 
                metric = 'Kappa',
                control=rpart.control(minsplit=2, xval=0),
                tuneGrid = data.frame(cp=c(0.0001, 0.0005, 0.001, 0.0015)),
                # tuneLength = 20,
                trControl = trc)

tune.bal
plot(tune.bal)
# now for the best tree
fit.tree = tune.bal$finalModel
plot(fit.tree, uniform=TRUE, branch=0.5, 
     main="Classification Tree for Marketing", margin=0.1)
text(fit.tree,  use.n=F, pretty=F, cex=0.6)

nrow(fit.tree$frame)

# now finally check on the test set using test data
pred.bal = predict(tune.bal, bm.test, type='raw')
cm.out = confusionMatrix(pred.bal, bm.test$y, positive='yes')
cm.out
# kappa 0.3814  recall 0.64162  precision 0.37248
profit(cm.out$table) # $67,300

#
# random forest
#
trc <- trainControl(method = "repeatedcv",
                    number = 10,
                    repeats = 5,
                    classProbs = TRUE,
                    allowParallel = T)
grid = expand.grid(mtry = c(5,6,7))
set.seed(1)    # for better comparison
tune.rf <- train(y ~ ., data=bal.train, 
                  method = "rf", 
                  metric = 'Kappa',
                  control=rpart.control(minsplit=2, xval=0),
                  tuneGrid = grid,
                  trControl = trc)

tune.rf
plot(tune.rf)

# now finally check on the test set using test data
pred.rf = predict(tune.rf, bm.test, type='raw')
cm.out = confusionMatrix(pred.rf, bm.test$y, positive='yes')
cm.out

profit(cm.out$table) # $80,100


