# read in the orange small data set
# 50000 obs 230 vars
kdd2009.data <- read.delim(file.choose(), na.strings=c('',NA))
summary(kdd2009.data)
# 190 numeric 40 categorical

#because the data set is so complex, 
#  we are going to build a dataset 'coldata' with feature properties
coldata <- data.frame(coltype=sapply(kdd2009.data, class))
summary(coldata)   # this is waht R thinks of the data
# let us visualize where these features are in the dataset
library(ggplot2)

ggplot(coldata, aes(x=1:nrow(coldata) , y=coltype)) + geom_point(aes(color=coltype)) +
  theme(text = element_text(size=20)) +
  labs(title = "Feature Types", x="Feature Number", y="Feature Type")

# data  has been anonymized and so the text strings do not make any sense
library(plyr)
count(kdd2009.data, "Var193")

# lots of missing data
# numeric
count(kdd2009.data, "Var1")   # 49,298 missing!  And, very few unique values
#factor
count(kdd2009.data, "Var191")

# let us assess the missingness first
# add how many missing column
coldata['num.na'] <- sapply(kdd2009.data, function(x) sum(is.na(x)))
summary(coldata$num.na)
# so the median of number missing is 48,510, and there is at least one variable with no data!!

# let us visualize this
ggplot(coldata, aes(num.na)) + geom_histogram(binwidth=100) +
  theme(text = element_text(size=20)) +
  labs(title="Histogram of Features by Number of Missing Values", 
       x="Number of Missing Values", y="Number of Features")

# looking at the right end
ggplot(coldata[coldata$num.na>45000,], aes(num.na)) + geom_histogram(binwidth=25) +
  theme(text = element_text(size=20)) +
  labs(title="Detail of Histogram of Missingness", 
       x="Number of Missing Values", y="Number of Features")
# 18 columns are empty!!

# now let us turn to the number of unique values
#add the number of unique values (other than missing) to it
coldata['num.unique'] <- apply(kdd2009.data, 2, function(x)length(unique(na.omit(x))))

summary(coldata['num.unique']) 
# the median column has only 38 unique values
# let us look at this more closely

ggplot(coldata, aes(num.unique)) + geom_histogram(binwidth=200) +
theme(text = element_text(size=20)) +
  labs(title="Histogram of Features by Number of Uniques", 
       x="Number of Uniques", y="Number of Features")

#let us look at the left part in great detail
ggplot(coldata[coldata$num.unique<=50,], aes(num.unique)) + geom_histogram(binwidth=1) +
theme(text = element_text(size=20)) +
  labs(title="Detail of Histogram of Number of Uniques", 
       x="Number of Uniques", y="Number of Features")

# so there are 18 features with no data and 5 with a single value!!

coldata[coldata$num.unique <=1,]

# let us look at a couple
count(kdd2009.data$Var118)  #numeric
count(kdd2009.data$Var191)  #factor

# the columns with very few different values have near zero variance\
#  NZV cause problems with most models
# library caret has a function to detect this
library(caret)
nzv <- nearZeroVar(kdd2009.data)
nzv
length(nzv)  

count(kdd2009.data$Var2)
# caret thinks that 84 of the variables have near zero variance
# Remove the empty columns 
kdd <- kdd2009.data[,-nzv]

# now let us recompute col data into coldata
coldata <- data.frame(coltype=sapply(kdd, class))
coldata['num.na'] <- sapply(kdd, function(x) sum(is.na(x)))
coldata['num.unique'] <- apply(kdd, 2, function(x)length(unique(na.omit(x))))

#now we would like to find the variables that are highly correlated
# let us look at numeric variables first
numcols = rownames(coldata[coldata$coltype!='factor',])
length(numcols)  # 118 variables

tick = proc.time()
corr = cor(kdd[,numcols], method='spearman', use='pairwise')
proc.time() - tick  # 106 seconds

findCorrelation(corr, verbose=T, names=T, cutoff=0.95)
# correlated columns are:
# (10, 11), (4, 34, 96), (36, 50, 89), (35, 60, 61), (48, 75, 86)  
# let us check

pairs(kdd[,c(10,11)])  # Var21 and Var22 are very correlated
pairs(kdd[,c(4,34,96)]) # Var9, Var66, Var156 also are very correlated
pairs(kdd[,c(36,50,89)]) # Var71, Var91, Var148 are also very correlated
pairs(kdd[,c(35,60,61)])  # Var68, Var104, Var105 are also very correlated
pairs(kdd[,c(48,75,86)])  # Var88, Var128, and Var145 are very correlated

# let us find the names of columns to drop
dropcols = c('Var22', 'Var66', 'Var156', 'Var91', 'Var148', 'Var104', 'Var105', 'Var128','Var145')

# now let us look at factors
faccols = rownames(coldata[coldata$coltype=='factor',])
length(faccols)  # 28 variables
faccols
combos <- combn(faccols,2) 
ncol(combos)  # 378 pairs

# chi-sq test of independence
# try a sample first
tab <- table(kdd[,c('Var192','Var193')])
chisq.test(tab)
ggplot(kdd, aes(Var192, Var193)) + geom_jitter() # no relationship revealed

# do not do, will take over 10 minutes
tick = proc.time()
for (i in 1:ncol(combos)) {
  # cat(combos[1,i],' ',combos[2,i],'\n')
  tab <- table(na.omit(kdd[,combos[,i]]))
  pval <- chisq.test(tab)$p.value
  if (!is.nan(pval) & pval < 0.01) cat(combos[1,i],' ',combos[2,i],' pvalue ',pval, '\n')
}
proc.time() - tick # 10 minutes

#now for a better measure Proportional Reduction in Error
library(rapportools)
# try a sample first
tab <- table(kdd[,c('Var192','Var193')])
lt <- lambda.test(tab)
pre <- min(c(lt$row, lt$col))
pre   # 0.002 not really related!

for (i in 1:ncol(combos)) {
  # cat(combos[1,i],' ',combos[2,i],'\n')
  tab <- table(kdd[,combos[,i]])
  lt <- lambda.test(tab)
  pre <- min(c(lt$row, lt$col))
  if (pre > 0.8) cat(combos[1,i],' ',combos[2,i],' pre ',pre, '\n')
}

# the significant relationships are (198, 220, 222), (200, 214), (207, 227)
# let us look at the scatter plots
pairs(kdd[,c('Var198','Var220','Var222')])
pairs(kdd[,c('Var200','Var214')])
pairs(kdd[,c('Var207','Var227')])
# none of them are that related!!  (missing value pattern is similar)

# so we just have dropcols to remove
kdd <- kdd[,!(names(kdd) %in% dropcols)]

# now read in the labels
# churn -1 = customer stays, 1 = customer leaves
# read in Data/KDD2009/orange_small/orange_small_train.churn.labels.csv
lc <- read.delim(file.choose(), header=F)
hist(lc$V1)

# appetency -1 = customer does not buy additional unrelated services, 1 = customer buys such services
# read in Data/KDD2009/orange_small/orange_small_train.appetency.labels.csv
la <- read.delim(file.choose(), header=F)
summary(la)

# upselling -1 = customer does not buy additional related services, 1 = customer buys such services
# read in Data/KDD2009/orange_small/orange_small_train.upselling.labels.csv
lu <- read.delim(file.choose(), header=F)
summary(lu)

# now make a new dataframe labels
labels <- data.frame(churn=lc$V1, app=la$V1, up=lu$V1)
#churn
labels$churn <- (1+ labels$churn)/2  # now 0 for staying, 1 for leaving
summary(labels$churn)
labels$churnf <- as.factor(ifelse(labels$churn==0, 'no', 'yes'))
summary(labels$churnf)
#app
labels$app <- (1+ labels$app)/2  # now 0 no app, 1 for yes app
summary(labels$app)
labels$appf <- as.factor(ifelse(labels$app==0, 'no', 'yes'))
summary(labels$appf)
#up
labels$up <- (1+ labels$up)/2  # now 0 no up, 1 for yes up
summary(labels$up)
labels$upf <- as.factor(ifelse(labels$up==0, 'no', 'yes'))
summary(labels$upf)

#now let us create a vector of row numbers to pick the test set
set.seed(2345)
# use 20% only for testing, do not calibrate, or tune your model using this
test <- sample(1:nrow(kdd), 0.2*nrow(kdd), replace=F)

# save kdd, labels, test 
#      into "~/Dropbox/Documents/CLASS/CIS417/Data/KDD2009/kdd.Rda"
save(kdd, labels, test, file=file.choose())

