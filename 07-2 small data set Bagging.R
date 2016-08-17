#
# SMALL DATA SET -- BOOTSTRAPPING AND BAGGING
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

# multicore processing (MACs only)
library(doMC)
registerDoMC(cores = 2)

# Now set up training and testing data sets
# we will a very tiny test set
set.seed(432) # for reproduceable results
# we will use createDataPartition from caret as it creates a sample while preserving
# the class distribution, i.e., it samples within each class
train <- createDataPartition(y=bm$ira,  # creates a stratified sample within class
                             p = 0.2,  # proportion for training
                             list=F)     #put result in a vector
bm.train <- bm[train,]
bm.test <- bm[-train,]

prop.table(table(bm.train$ira))

# first cross-validation with a tree model 
trc <- trainControl(method = "repeatedcv",
                    number = 10,
                    repeats = 5,
                    classProbs = TRUE,
                    allowParallel = T)
set.seed(1)    # for better comparison
tune.cv = train(ira ~ ., data=bm.train, 
                       method = "rpart", 
                       metric = 'Accuracy',
                       control=rpart.control(minsplit=2, xval=0),
                       tuneLength = 20,
                       trControl = trc)

tune.cv
plot(tune.cv)

results.cv = tune.cv$results

# now for bootstrapping with tree model
trc <- trainControl(method = "boot",
                    number = 10,
                    repeats = 5,
                    classProbs = TRUE,
                    allowParallel = T)
set.seed(1)
tune.b = train(ira ~ ., data=bm.train, 
                      method = "rpart", 
                      metric = 'Accuracy',
                      control=rpart.control(minsplit=2, xval=0),
                      tuneLength = 20,
                      trControl = trc)

tune.b
plot(tune.b)


results.b = tune.b$results

ggplot(results.cv, aes(cp, Accuracy)) + geom_line(color='red') +
  geom_line(aes(results.b$cp, results.b$Accuracy), color='blue') +
  geom_line(aes(results.cv$cp, results.cv$AccuracySD), color='red', lty=2) +
  geom_line(aes(results.b$cp, results.b$AccuracySD), color='blue', lty=2) +
  theme(text= element_text(size=20)) 

# some more processing of the CV result

# now for the best tree
fit.tree = tune.cv$finalModel
plot(fit.tree, uniform=TRUE, branch=0.5, 
     main="Classification Tree for Marketing", margin=0.1)
text(fit.tree,  use.n=F, pretty=F, cex=0.6)

nrow(fit.tree$frame)

# now finally check on the test set using test data
pred.cv = predict(tune.cv, bm.test, type='raw')
confusionMatrix(pred.cv, bm.test$ira, positive='YES')

#
# BAGGING using adabag
#

library(adabag)
set.seed(1)
ira.b = bagging(ira~., data=bm.train, mfinal=100, 
                   control=rpart.control(minsplit=2, xval=0))

summary(ira.b)

pred.b = predict.bagging(ira.b, bm.test, newfinal=100)
confusionMatrix(pred.b$class, bm.test$ira, positive='YES')

num.nodes = c()
for (i in 1:100) num.nodes = append(num.nodes, nrow(ira.b$trees[[i]]$frame))

library(ggplot2)
ggplot(as.data.frame(num.nodes), aes(num.nodes)) + geom_histogram(binwidth=1)

