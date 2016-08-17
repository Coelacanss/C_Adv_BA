### This script is for Advanced BA course final project - TEAM 3
### Part 2: extract new features from serving log data, and to clean it, in order to
### create enough new features as the supplement to users profile data.

library(caret)
library(lubridate)
library(xgboost)
library(readr)
library(stringr)
library(car)
library(plyr)

## load serving log data
load(file = 'sessions.Rds')
summary(session)
## check near zero variables
nearZeroVar(session)

## look at missing data, and select data to be supplement
sum(session$action_detail == '')  # 1,126,204 NA/ 10.66%
length(levels(session$action_detail)) # 156 levels
sum(session$action == '') # 15,22% NA
length(levels(session$action)) # 360 levels
sum(session$action_type == '') # 10.66% NA
length(levels(session$action_type)) # 11 levels
sum(session$device_type == '') # no NA
length(levels(session$device_type)) # 14 levels
## the variable "action_detail" has proper proportion of missing data, while
## it has enough levels to be supplement

## extract user list
session$user_id = as.character(session$user_id)
usersList = unique(session$user_id)

# load previous supplment data
# load(file = 'data.s.Rda') 

count = 0

time.now = proc.time()
for (username in usersList) {
      
      # for tracking
      count = count + 1
      print(count)
      
      # get subset of a user
      user.current = NULL
      user.current <- session[session$user_id == username,]
      newFeatures = NULL
      
      # extract "action_detail"
      action_detail = NULL
      action_detail = as.data.frame(table(user.current$action_detail))
      rownames(action_detail) = action_detail[,1]
      rownames(action_detail)[1] = 'actionNA'
      rownames(action_detail)[2] = 'actionUnknown'
      action_detail[,1] = NULL
      newFeatures$id = user.current[1,1]
      newFeatures = as.data.frame(newFeatures)
      newFeatures = cbind(newFeatures, t(action_detail))
      
      # extract "secs_elapsed", to create 3 new features: secs_min, secs_max, secs_mean
      secs = NULL
      secs_min = min(user.current$secs_elapsed, na.rm = T)
      secs_max = max(user.current$secs_elapsed, na.rm = T)
      secs_mean = mean(user.current$secs_elapsed, na.rm = T)
      secs = cbind(secs_min, secs_max, secs_mean)
      newFeatures = cbind(newFeatures, secs)
      
      # extrat the number of actions of a user, as a new feature
      num_actions = nrow(user.current)
      newFeatures = cbind(newFeatures, num_actions)
      
      # save these new features
      data.s = rbind(data.s, newFeatures)
}

proc.time() - time.now # it takes over 10 hours

## save data
save(data.s, file = 'data.s.Rda')  # 's' stands for suppliment
