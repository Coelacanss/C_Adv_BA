### This script is for Advanced BA course final project - TEAM 3
### Part 1: try random forest models on only users profile data

library(caret)
library(lubridate)
library(readr)
library(stringr)
library(caret)
library(car)
library(randomForest)
library(e1071)

## create function to compute ndcg-5, as the metric in training process
ndcg5 <- function(preds, dtrain) {
      
      labels <- getinfo(dtrain,"label")
      num.class = 12
      pred <- matrix(preds, nrow = num.class)
      top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
      
      x <- ifelse(top==labels,1,0)
      dcg <- function(y) sum((2^y - 1)/log(2:(length(y)+1), base = 2))
      ndcg <- mean(apply(x,1,dcg))
      return(list(metric = "ndcg5", value = ndcg))
}

## load user profile data
load(file = 'dataAirbnb.Rda') # 213,451 obs, 16 variables
## check class variable
table(dataAirbnb$country_destination) # very imbalanced
# NDCG-5 benchmark:0.68411

## create training user list and testing user list
set.seed(12345)
inTrain = sample(1:nrow(dataAirbnb), 0.7*nrow(dataAirbnb))
training = dataAirbnb[inTrain,'id']
testing = dataAirbnb[-inTrain,'id']
 # save(training, 'training.Rda')
 # save(testing, 'testing.Rda')

## get class variable
classes = training$country_destination
training$country_destination = NULL
id.test = testing$id

##########################################################################################
## simple model: 149,416obs(70%) & 16 variables
##
## Processing & Feature engineering
# combine training and testing data
data.all = rbind(training, testing)
# remove date_first_booking
data.all$date_first_booking = NULL
# convert date_account_created into details(year, month, day, weekday)
data.all$date_account_created = as.Date(as.character(data.all$date_account_created), 
                                        format = '%Y-%m-%d')
data.all$dac_year = as.factor(year(data.all$date_account_created))
data.all$dac_month = as.factor(month(data.all$date_account_created))
data.all$dac_day = as.factor(day(data.all$date_account_created))
data.all$dac_weekday = as.factor(weekdays(data.all$date_account_created))
data.all$date_account_created = NULL
# extract year from timestamp_first_active
data.all$tfa_year = substring(as.character(data.all$timestamp_first_active), 1, 4)
data.all$tfa_year = as.factor(data.all$tfa_year)
data.all$timestamp_first_active = NULL
# remove NA data and data those against rules in "age" (younger than 14 or older than 100)
data.all[is.na(data.all$age), 'age'] = -3
data.all[(data.all$age < 14 & data.all$age != -3) | data.all$age > 100, 'age'] <- -1
# cut age into some levels, in order to alleviate the computation
data.all$age = cut(data.all$age, breaks = c(-4,-2,0,21,25,30,35,40,45,50,55,60,65,70,80,Inf))

# others (in order to alleviate the computation)
data.all$language = NULL
data.all$first_browser = NULL
data.all$signup_flow = as.factor(data.all$signup_flow)

# split data into training and testing subset
data.train = data.all[(data.all$id %in% training),]
data.test = data.all[data.all$id %in% testing,]
data.train$country_destination = as.factor(as.character(classes))

# train model.rf
set.seed(123)
time.now = proc.time()
model.rf <- randomForest(country_destination ~., data = data.train[,-1],
                         ntree = 200, do.trace = TRUE)
proc.time() - time.now # takes over 1 hour

## prediction
pred <- predict(model.rf, data.test[,-1], type = 'prob')
pred.top5 <- as.vector(apply(pred, 1, function(x) names(sort(x)[12:8])))
# NDCG-5 Score: 0.778

####################################################################################
## sophisticated model: 20,000obs(70%) & 61 variables
##
## Processing & Feature engineering
# combine training and testing data
data.all = rbind(training, testing)
# remove date_first_booking
data.all$date_first_booking = NULL
# convert date_account_created into details(year, month, day, weekday)
data.all$date_account_created = as.Date(as.character(data.all$date_account_created), 
                                        format = '%Y-%m-%d')
data.all$dac_year = as.factor(year(data.all$date_account_created))
data.all$dac_month = as.factor(month(data.all$date_account_created))
data.all$dac_day = as.factor(day(data.all$date_account_created))
data.all$dac_weekday = as.factor(weekdays(data.all$date_account_created))
data.all$date_account_created = NULL
# extract year from timestamp_first_active
data.all$tfa_year = substring(as.character(data.all$timestamp_first_active), 1, 4)
data.all$tfa_year = as.factor(data.all$tfa_year)
data.all$timestamp_first_active = NULL
# remove NA data and data those against rules in "age" (younger than 14 or older than 100)
data.all[is.na(data.all$age), 'age'] = -3
data.all[(data.all$age < 14 & data.all$age != -3) | data.all$age > 100, 'age'] <- -1
# convert into factors
data.all$signup_flow = as.factor(data.all$signup_flow)

## one-hot encoding, to create dummy dataframe
ohe.feats = c('gender', 'signup_method', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type')
dummies <- dummyVars(~ gender + signup_method + affiliate_channel + affiliate_provider + first_affiliate_tracked + signup_app + first_device_type, data = data.all)
df.ohe <- as.data.frame(predict(dummies, newdata = data.all))
data.all <- cbind(data.all[,-c(which(colnames(data.all) %in% ohe.feats))], df.ohe)

# split data into training and testing subset
# subseting 20000 obs in order to alleviate compution
data.train = data.all[(data.all$id %in% training),]
InSample = sample(1:nrow(data.train), 20000)
data.train = data.train[inTrain,]
data.test = data.all[data.all$id %in% testing,]
data.train$country_destination = as.factor(as.character(classes))

## parameters tuning using 2000 obs
InTuning = sample(1:nrow(data.train), 2000)
data.tuning = data.train[InTuning,]
grid <- expand.grid(mtry = c(7,8,9),
                    ntree = c(100, 200, 300))
                    
set.seed(123) 
tune.log = train(country_destination ~ ., data=data.tuning[,-1], 
                 method = "rf", 
                 tuneGrid = grid,
                 metric = ndcg5)
tune.log$bestTune # mtry = 8, ntree = 200

## train model.rf
set.seed(123)
time.now = proc.time()
model.rf = train(country_destination ~ ., data=data.tuning[,-1], 
                 method = "rf", 
                 mtry = 8,
                 ntree = 200,
                 metric = ndcg5)
proc.time() - time.now # takes over 3 hours

## prediction
pred <- predict(model.rf, data.test[,-1], type = 'prob')
pred.top5 <- as.vector(apply(pred, 1, function(x) names(sort(x)[12:8])))
# NDCG-5 Score: 0.828

