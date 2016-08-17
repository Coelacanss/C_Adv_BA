### Airbnb Project on Kaggle

library(caret)
library(lubridate)

# Check stats
load(file = 'dataAirbnb.Rda'); load(file = 'dataTest.Rda')

dataAirbnb <- read.csv(file = 'train_users_2.csv', stringsAsFactors = FALSE, 
                       quote = "\"", skipNul = TRUE)
dataTest <- read.csv(file = 'test_users.csv', stringsAsFactors = FALSE, skipNul = TRUE)
set.seed(1111)
inSample = sample(nrow(dataAirbnb), 20000)
dataSample = dataAirbnb[inSample,]
# save(dataSample, file = 'dataSample.Rda')
# load(file = 'dataSample.Rda')

# Function dataClean
dataClean <- function(x){
      x$id = NULL
      x$date_first_booking = NULL
      
#      x$date_account_created = as.Date(as.character(x$date_account_created), format = '%Y-%m-%d')
#      x$date_account_created = paste0(year(x$date_account_created), '-', month(x$date_account_created))
#      x$date_account_created = as.factor(x$date_account_created)
      
      x$date_account_created = as.Date(as.character(x$date_account_created), format = '%Y-%m-%d')
      x$dac_year = as.factor(year(x$date_account_created))
      x$dac_month = as.factor(month(x$date_account_created))
      x$dac_day = as.factor(day(x$date_account_created))
      # maybe add 'weekday'?
      x$date_account_created = NULL
      
      x[is.na(x$age), 'age'] = -2
      x[(x$age < 14 | x$age > 95) & x$age != -2, 'age'] = -1
      x$age = cut(x$age, breaks = c(-3,-1,21,25,30,35,40,45,50,55,60,65,70,80,Inf))
      
      x$timestamp_first_active = cut(x$timestamp_first_active, 
                                     breaks = c(2.000e+13, 2.012e+13, 2.013e+13, 2.014e+13, 2.020e+13))
      x$language = NULL # english speaker intends to choos US, this may induce error
      
      x$signup_method = as.character(x$signup_method)
      x[(x$signup_method != 'basic') & (x$signup_method != 'facebook') & (x$signup_method != 'google'), 
        'signup_method'] = 'basic'
      x$signup_method = as.factor(x$signup_method)
      
      x[!(x$affiliate_provider %in% levels(x$affiliate_provider)), 'affiliate_provider'] = 'other'
      
      x$first_browser = NULL
      
      return(x)
}

# Build training set, testing set, and validation set.
set.seed(12345)
inBuild = createDataPartition(y = dataSample$country_destination, p = 0.7, list = FALSE)
validation = dataSample[-inBuild,]
buildData = dataSample[inBuild,]
set.seed(22345)
inTrain = createDataPartition(y = buildData$country_destination, p = 0.7, list = FALSE)
training = buildData[inTrain,]
testing = buildData[-inTrain,]
rm(buildData); rm(inBuild); rm(inTrain)

data.train = dataClean(training)
data.test = dataClean(testing)
data.validation = dataClean(validation)
dataTest = dataClean(dataTest)

# Build Model
set.seed(1111)
time.now = proc.time()
model.rf = train(country_destination ~., method = "rf", 
                 data = data.train, ntree = 50, do.trace = TRUE)
proc.time() - time.now
# save(model.rf, file = 'model.rf.Rds')
# load(file = 'model.rf.Rds')

# Testing accuracy
pred.rf = predict(model.rf, data.test, type = 'raw')
confusionMatrix(data.test$country_destination, pred.rf)$overall[1]
confusionMatrix(data.test$country_destination, pred.rf)
data.test$result = as.character(pred.rf)

## Predict
pred.rf = predict(model.rf, dataTest, type = 'prob')
predictions_top5 <- as.vector(apply(pred.rf, 1, function(x) names(sort(x)[12:8])))

## create submission 
ids <- NULL
dataTest$id = as.character(dataTest$id)
for (i in 1:nrow(dataTest)) {
      idx <- dataTest$id[i]
      ids <- append(ids, rep(idx,5))
}
submission <- NULL
submission$id <- ids
submission$country <- predictions_top5

submission <- as.data.frame(submission)
write.csv(submission, "submissionRF.csv", quote=FALSE, row.names = FALSE)

# Group residual data other than NDF and US
data.train = dataClean(training)
data.train = data.train[((data.train$country_destination != 'NDF') & 
                        (data.train$country_destination != 'US')), ]
data.train$country_destination = as.factor(as.character(data.train$country_destination))

# Build model.residual
set.seed(1111)
time.now = proc.time()
model.rf.residual = train(country_destination ~., method = "rf", 
                          data = data.train, ntree = 20, do.trace = TRUE)
proc.time() - time.now
# save(model.rf.residual, file = 'model.rf.residual2.Rds')
# load(file = 'model.rf.residual')

# Test residual accuracy
data.test = dataClean(testing)
data.test = data.test[((data.test$country_destination != 'NDF') &
                        (data.test$country_destination != 'US')), ]
data.test$country_destination = as.factor(as.character(data.test$country_destination))
data.test = data.test[((data.test$date_first_booking != '2010-2') &
                        (data.test$date_first_booking != '2015-6')), ]
pred.rf.residual = predict(model.rf.residual, data.test)
confusionMatrix(data.test$country_destination, pred.rf.residual)$overall[1]  #0.381



### Result test
testing$result = pred.rf
testing.residual = testing[(testing$country_destination != 'NDF'), c(1:14)]
pred.redidual = predict(model.rf.residual, 
                        testing.residual[(testing.residual$date_first_booking != '2010-2' &
                                          testing.residual$date_first_booking != '2015-6'),])
testing[(testing$country_destination != 'NDF'), 15] = 'US'

confusionMatrix(testing$country_destination, testing$result)$overall[1]
confusionMatrix(testing$country_destination, testing$result)

### Short cuts
sapply(data.train, function(x){sum(is.na(x))})

