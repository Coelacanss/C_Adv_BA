#
# R E S A M P L I N G   F O R   M O D E L   T U N I N G   A N D    C O M P A R I S O N
#

# load bank IRA data
bm <- read.csv("bank-data.csv", row.names=1)
bm <- read.csv(file.choose(), row.names=1)
# load the caret library
library(caret)

# multicore processing
library(doParallel)
# find how many independent threads you can have
makeCluster(detectCores())
# set up to use multiple cores
registerDoParallel(cores=2)
?train
# multicore processing (MACs only)
library(doMC)
registerDoMC(cores = 2)

# Now set up training and testing data sets
set.seed(432) # for reproduceable results
# we will use createDataPartition from caret as it creates a sample while preserving
# the class distribution, i.e., it samples within each class
train <- createDataPartition(y=bm$ira,  # creates a stratified sample within class
                             p = 0.6667,  # proportion for training
                             list=F)     #put result in a vector
bm.train <- bm[train,]
bm.test <- bm[-train,]

prop.table(table(bm$ira))
prop.table(table(bm.train$ira))

# first cross-validation with a tree model (8 secs)
trc <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5,
                           classProbs = TRUE,
                           allowParallel = T)
set.seed(1)    # for better comparison
tune.cv.tree = train(ira ~ ., data=bm.train, 
             method = "rpart", 
             metric = 'Accuracy',
             control=rpart.control(minsplit=2, xval=0),
             tuneLength = 20,
             trControl = trc)

tune.cv.tree
plot(tune.cv.tree)

results.cv = tune.cv.tree$results  # what is shown in the summary
# now for the best tree
fit.tree = tune.cv.tree$finalModel
plot(fit.tree, uniform=TRUE, branch=0.5, 
     main="Classification Tree for Marketing", margin=0.1)
text(fit.tree,  use.n=F, pretty=F, cex=0.6)

# now for bootstrapping with tree model
trc <- trainControl(method = "boot",
                    number = 10,
                    repeats = 5,
                    classProbs = TRUE,
                    allowParallel = T)
set.seed(1)
tune.b.tree = train(ira ~ ., data=bm.train, 
                  method = "rpart", 
                  metric = 'Accuracy',
                  control=rpart.control(minsplit=2, xval=0),
                  tuneLength = 20,
                  trControl = trc)

tune.b.tree
plot(tune.b.tree)


results.b = tune.b.tree$results

ggplot(results.cv, aes(cp, Accuracy)) + geom_line(color='red') +
  geom_line(aes(results.b$cp, results.b$Accuracy), color='blue') +
  geom_line(aes(results.cv$cp, results.cv$AccuracySD), color='red', lty=2) +
  geom_line(aes(results.b$cp, results.b$AccuracySD), color='blue', lty=2) 


#
# now for SVM
#

trc <- trainControl(method = "repeatedcv",
                    number = 10,
                    repeats = 5,
                    allowParallel = T)

set.seed(1)    # for better comparison
tune.svm.rbf = train(ira ~ ., data=bm.train, 
                     method = "svmRadial", 
                     tuneLength = 9,   # 2^-1, 2^0, 2^1, ..., 2^7
                    #tuneGrid = svmTuneGrid,  
                     preProc = c("center", "scale"),
                     metric = 'Accuracy',
                     trControl = trc)

tune.svm.rbf
plot(tune.svm.rbf)

# now fine tune
grid <- expand.grid(sigma = c(.01, .06, 0.1),
                    C = c(4, 6, 8, 10, 12)
)
trc <- trainControl(method = "repeatedcv",
                    number = 10,
                    repeats = 5,
                    allowParallel = T)

set.seed(1)    # for better comparison
tune.svm.rbf = train(ira ~ ., data=bm.train, 
                    method = "svmRadial", 
                    tuneGrid = grid,  
                    preProc = c("center", "scale"),
                    metric = 'Accuracy',
                    trControl = trc)

tune.svm.rbf$bestTune
plot(tune.svm.rbf)

# now let us try a different kernel - polynomial
#  http://topepo.github.io/caret/Support_Vector_Machines.html
tick = proc.time()
trc <- trainControl(method = "repeatedcv",
                    number = 10,
                    repeats = 5,
                    allowParallel = T)

set.seed(1)    # for better comparison
tune.svm.poly = train(ira ~ ., data=bm.train, 
                     method = "svmPoly", 
                     tuneLength = 4,
                     preProc = c("center", "scale"),
                     metric = 'Accuracy',
                     trControl = trc)
proc.time() - tick
tune.svm.poly

#
# and now for linear kernel
#


trc <- trainControl(method = "repeatedcv",
                    number = 10,
                    repeats = 5,
                    allowParallel = T)
grid <- expand.grid(C = 2^(-7:0))
tick = proc.time()
set.seed(1)    # for better comparison
tune.svm.lin = train(ira ~ ., data=bm.train, 
                      method = "svmLinear", 
                      tuneGrid = grid,
                      preProc = c("center", "scale"),
                      metric = 'Accuracy',
                      trControl = trc)
proc.time() - tick
tune.svm.lin
plot(tune.svm.lin)

#
# logistic regression
# see http://topepo.github.io/caret/Logistic_Regression.html
#
trc <- trainControl(method = "repeatedcv",
                    number = 10,
                    repeats = 5,
                    allowParallel = T)

set.seed(1)    # for better comparison
tune.log = train(ira ~ ., data=bm.train, 
                     method = "glm", 
                     family = 'binomial',
                     preProc = c("center", "scale"),
                     metric = 'Accuracy',
                     trControl = trc)

tune.log

summary(tune.log)
# note how the formula interface of caret and glm 
# makes binary variables out of factors
# factors with k levels make k-1 binary variables

#
# Using resamples() to compare models
#
rValues <- resamples(list(CART=tune.cv.tree,svmrbf=tune.svm.rbf, 
                          svmpoly=tune.svm.poly, svmlin=tune.svm.lin,
                          logistic=tune.log))
summary(rValues)

bwplot(rValues,metric="Accuracy")	

dotplot(rValues, metric="Accuracy")

# now finally check on the test set using test data
pred.svm.poly = predict(tune.svm.poly, bm.test, type='raw')
confusionMatrix(pred.svm.poly, bm.test$ira, positive='YES')