### This script is for Advanced BA course final project - TEAM 3
### Part 3: combine users profile data and suppliment data (extracted features from
### serving log data), build and tune xgboost model.

library(caret)
library(lubridate)
library(xgboost)
library(readr)
library(stringr)
library(car)
library(plyr)
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

# load data
load(file = 'training.Rda'); load(file = 'testing.Rda')
load(file = 'dataTest.Rda'); load(file = 'data.s.Rda')

training$id = as.character(training$id)
data.s$id = as.character(data.s$id)
dataTest$id = as.character(dataTest$id)

# get class variable
classes = training$country_destination
training$country_destination = NULL
id.test = testing$id

## Processing & Feature engineering
# combine training and testing data
data.all = rbind(training, testing)
# remove date_first_booking
data.all$date_first_booking = NULL
# convert NA into -1
data.all[is.na(data.all)] <- -1

# date_account_created
data.all$date_account_created = as.Date(as.character(data.all$date_account_created), 
                                        format = '%Y-%m-%d')
 # data.all$dac_year = as.factor(year(data.all$date_account_created))
data.all$dac_month = as.factor(month(data.all$date_account_created))
data.all$dac_day = as.factor(day(data.all$date_account_created))
data.all$dac_weekday = as.factor(weekdays(data.all$date_account_created))
data.all$date_account_created = NULL

# timestamp_first_active
 # data.all$tfa_year = substring(as.character(data.all$timestamp_first_active), 1, 4)
 # data.all$tfa_year = as.factor(data.all$tfa_year)
data.all$timestamp_first_active = NULL

# age
data.all[is.na(data.all$age), 'age'] = -3
data.all[(data.all$age < 14 & data.all$age != -3) | data.all$age > 100, 'age'] <- -1

# others
 # data.all$language = NULL
 # data.all$first_browser = NULL
data.all$signup_flow = as.factor(data.all$signup_flow)

## one-hot encoding
ohe.feats = c('gender', 'age', 'signup_method', 'signup_flow', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type',
              'first_browser', 'language', 'dac_month', 'dac_day', 'dac_weekday')
dummies <- dummyVars(~ gender + age + signup_method + signup_flow + affiliate_channel + affiliate_provider + first_affiliate_tracked + signup_app + first_device_type +
                           first_browser + language + dac_month + dac_day + dac_weekday, data = data.all)
df.ohe <- as.data.frame(predict(dummies, newdata = data.all))
data.all.comb <- cbind(data.all[,-c(which(colnames(data.all) %in% ohe.feats))], df.ohe)
colnames(data.all.comb)[1] = 'id'

## combine suppliment data (serving log data)
data.all.comb = cbind(data.all.comb, data.s)
data.all.comb[,193] = NULL
unique(sapply(data.all.comb[,-1], class))
## clean NAs
data.all.comb[is.na(data.all.comb)] <- -1

## split data into training and testing
data.train = data.all.comb[(data.all.comb$id %in% training$id),]
data.test = data.all.comb[data.all.comb$id %in% id.test,]
labels <- recode(classes, "'NDF'=0; 'US'=1; 'other'=2; 'FR'=3; 'CA'=4; 'GB'=5; 'ES'=6; 'IT'=7; 'PT'=8; 'NL'=9; 'DE'=10; 'AU'=11")
data.labels = as.character(labels)

## train xgboost model
time.now = proc.time()
xgb <- xgboost(data = data.matrix(data.train[,-1]), 
               label = data.labels, 
               eta = 0.07,
               max_depth = 6, 
               nround = 200, 
               subsample = 0.6,
               colsample_bytree = 0.6,
               eval_metric = ndcg5,
               objective = "multi:softprob",
               num_class = 12,
               nthread = 3
)
proc.time() - time.now
# we tuned parameters manually since 'xgboost' seems not be included in 'train' function
# of caret package

## prediction
pred <- predict(xgb, data.matrix(data.test[,-1]))
pred <- as.data.frame(matrix(pred, nrow = 12))
rownames(pred) <- c('NDF','US','other','FR','IT','GB','ES','CA','DE','NL','AU','PT')
pred.top5 <- as.vector(apply(pred, 2, function(x) names(sort(x)[12:8])))
## best NDCG-5 Score: 0.87999

## save prediction
ids <- NULL
data.test$id = as.character(data.test$id)
for (i in 1:nrow(data.test)) {
      idx <- data.test$id[i]
      ids <- append(ids, rep(idx,5))
}
submission <- NULL
submission$id <- ids
submission$country <- pred.top5
submission.part1 <- as.data.frame(submission)

save(submission.part1, file = 'submission.part1.Rda')

load(file = 'submission.part2.Rda')
submission = rbind(submission.part1, submission.part2)
write.csv(submission, "submission68.csv", quote=FALSE, row.names = FALSE)

## check results
result1 = c()
list1 = seq(from = 1, to = 310476, by = 5)
for (i in list1) {
      result1 = c(result1, submission[i,'country'])
}
count(result1)

result2 = c()
list2 = seq(from = 2, to = 310477, by = 5)
for (i in list2) {
      result2 = c(result2, submission[i,'country'])
}
count(result2)

result3 = c()
list3 = seq(from = 3, to = 310478, by = 5)
for (i in list3) {
      result3 = c(result3, submission[i,'country'])
}
count(result3)

result4 = c()
list4 = seq(from = 4, to = 310479, by = 5)
for (i in list4) {
      result4 = c(result4, submission[i,'country'])
}
count(result4)

result5 = c()
list5 = seq(from = 5, to = 310480, by = 5)
for (i in list5) {
      result5 = c(result5, submission[i,'country'])
}
count(result5)
