### Airbnb Project on Kaggle

library(caret)
library(lubridate)
library(readr)
library(stringr)
library(caret)
library(car)

## load data
load(file = 'dataAirbnb.Rda')
table(dataAirbnb$country_destination) # very imbalanced
## NDCG-5 benchmark:65%

## create training user list and testing user list
set.seed(12345)
inTrain = sample(1:nrow(dataAirbnb), 0.7*nrow(dataAirbnb))
training = dataAirbnb[inTrain,'id']
testing = dataAirbnb[-inTrain,'id']
 # save(training, 'training.Rda')
 # save(testing, 'testing.Rda')

## get labels
classes = training$country_destination
training$country_destination = NULL
id.test = testing$id

## Processing
# combine training and testing data
data.all = rbind(training, testing)
# remove date_first_booking
data.all$date_first_booking = NULL

## Feature engineering
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

# remove NA data and data those against rules in "age"
data.all[is.na(data.all$age), 'age'] = -3
data.all[(data.all$age < 14 & data.all$age != -3) | data.all$age > 100, 'age'] <- -1
# cut age into some levels, in order to alleviate the computation
data.all$age = cut(data.all$age, breaks = c(-4,-2,0,21,25,30,35,40,45,50,55,60,65,70,80,Inf))

# others (in order to alleviate the computation)
data.all$language = NULL
data.all$first_browser = NULL
data.all$signup_flow = as.factor(data.all$signup_flow)

## one-hot encoding, to create dummy dataframe
ohe.feats = c('gender', 'signup_method', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type')
dummies <- dummyVars(~ gender + signup_method + affiliate_channel + affiliate_provider + first_affiliate_tracked + signup_app + first_device_type, data = data.all)
df.ohe <- as.data.frame(predict(dummies, newdata = data.all))
data.all <- cbind(data.all[,-c(which(colnames(data.all) %in% ohe.feats))], df.ohe)

a## split data into training and testing subset
data.train = data.all[(data.all$id %in% training),]
data.test = data.all[data.all$id %in% testing,]
data.train$country_destination = as.factor(as.character(classes))

## train model.rf
set.seed(123)
time.now = proc.time()
model.rf <- randomForest(country_destination ~., data = data.train[,-1],
                          ntree = 200, do.trace = TRUE)
proc.time() - time.now # takes over 1 hour

## prediction
pred <- predict(model.rf, data.test[,-1], type = 'prob')
pred.top5 <- as.vector(apply(pred, 1, function(x) names(sort(x)[12:8])))
