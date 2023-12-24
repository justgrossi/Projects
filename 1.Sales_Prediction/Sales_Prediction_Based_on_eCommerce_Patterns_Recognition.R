library(tidyverse)
library(magrittr)
library(caret)
library(smotefamily)
library(fastDummies)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ROCR)
library(e1071)
library(corrplot)

# Set seed for reproducibility and acquire data
set.seed(1234)
df <- read.csv("online_shoppers_intention.csv", header = T, stringsAsFactors = F)
str(df)
sum(is.na(df))

# Check balance
prop.table(table(df$Revenue))

#Encode into binary
df$Weekend <- ifelse(df$Weekend=='TRUE', 1, 0)
df$Revenue <- ifelse(df$Revenue=='TRUE', 1, 0)


Month_levels <- levels(as.factor(df$Month))
VisitorType_levels <- levels(as.factor(df$VisitorType))

# Identify and print near-zero-variance variables
near_zero_vars <- nearZeroVar(df)
near_zero_var_names <- names(df)[near_zero_vars]

print(near_zero_var_names)

# Remove near-zero-variance variables from the data frame
df <- df[, -near_zero_vars]

# Create index and train vs test set
train_ind <- createDataPartition(df$Revenue, p = .8, 
                                 list = FALSE, 
                                 times = 1)
train <- df[train_ind, ]
test <- df[-train_ind, ]
test$Revenue <- as.factor(test$Revenue)

# One-hot encoding strings
dummyCols <- dummy_cols(train, select_columns = c('Month', 'VisitorType'))

#Eliminate original variables
train <- dummyCols[, -c(10, 15)]
train <- train %>% relocate(Revenue, .after = VisitorType_Returning_Visitor)

#Change data type prior to applying SMOTE
train <- train %>% mutate_all(as.numeric)

# Resolve class imbalance
smoted <- train %>% SMOTE(train$Revenue, K = 5)

# Get and prepare the data
dat <- smoted$data
dat$class <- NULL

# Check SMOTED values for Revenue
nrow(dat %>% filter(Revenue!=0) %>% filter(Revenue!=1))
dat$Revenue <- as.factor(dat$Revenue)

# Now the data set is balanced
prop.table(table(dat$Revenue))

# Check if values need to be rounded
columnNames <- colnames(cbind(dummyCols[,18:30], dummyCols['Weekend']))
rows <- list()
for (col in columnNames) {
  nRows <- dat %>% 
    filter(!!sym(col) != 0) %>% 
    filter(!!sym(col) != 1) %>%
    nrow()
  
  rows <- append(rows, list(nRows))
}

print(rows)

# Round values
dat[columnNames] <- dat[columnNames] %>% round(digits = 0)

# Sanity check
rows2 <- list()
for (col in columnNames) {
  nRows2 <- dat %>% 
    filter(!!sym(col) != 0) %>% 
    filter(!!sym(col) != 1) %>%
    nrow()
  
  rows2 <- append(rows2, list(nRows2))
}

print(rows2)

# Recode into original values
dat['Month']=0
dat['VisitorType']=0
listMonths <- levels(as.factor(test$Month))
listVisitorType <- levels(as.factor(test$VisitorType))
newNames <- c(colnames(dummyCols[,18:30]))

dat$Month <- ifelse(dat[,newNames[1]]==1, listMonths[1],
                       ifelse(dat[,newNames[2]]==1, listMonths[2],
                              ifelse(dat[,newNames[3]]==1, listMonths[3],
                                     ifelse(dat[,newNames[4]]==1, listMonths[4],
                                            ifelse(dat[,newNames[5]]==1, listMonths[5],
                                                   ifelse(dat[,newNames[6]]==1, listMonths[6],
                                                          ifelse(dat[,newNames[7]]==1, listMonths[7],
                                                                 ifelse(dat[,newNames[8]]==1, listMonths[8],
                                                                        ifelse(dat[,newNames[9]]==1, listMonths[9], listMonths[10])))))))))

dat$VisitorType <- ifelse(dat[,newNames[11]]==1, listVisitorType[1],
                    ifelse(dat[,newNames[12]]==1, listVisitorType[2], listVisitorType[3]))

# Eliminate one hot encoded variables
dat <- dat %>% select(-c(newNames))

# Get columns order from test set
test_columns <- colnames(test)

# Reorder train set to match
dat <- dat[, test_columns]

##### Train data are now ready
# Set Train Control
trControl <- trainControl(method = "cv", number = 10)

####### Random Forest #########
# Set grid to look for the best MTRY parameter:
tuneGrid <- expand.grid(.mtry = c(1: 16))

# Find the best value for mtry
mtryRf <- train(y=dat[, 17],    
                x=dat[, 1:16],
                data=dat,
                method = "rf",
                metric = "Accuracy",
                tuneGrid = tuneGrid,
                trControl = trControl,
                preProcess = c('center', 'scale'),
                importance = TRUE,
                ntree = 30)
# Visualize and store best value of MTRY:
print(mtryRf)
plot(mtryRf)
best_mtry <- mtryRf$bestTune$mtry
# The value corresponding to the highest in-sample Accuracy is 6


# Search best MAXNODE parameter:
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
  rf_maxnode <- train(y=dat[, 17],
                      x=dat[,1:16],
                      data=dat,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      preProcess = c('center', 'scale'),
                      importance = TRUE,
                      maxnodes = maxnodes,
                      ntree = 30)
  
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_maxnode <- resamples(store_maxnode)
summary(results_maxnode)
# The maxnode corresponding to the highest in-sample Accuracy is 15

# Search for best NTREE:
store_maxtrees <- list()
for (ntree in c(50, 100, 150, 200, 250)) {
  rf_maxtrees <- train(y=dat[, 17],
                       x=dat[,1:16],
                       data=dat,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       preProcess = c('center', 'scale'),
                       importance = TRUE,
                       maxnodes = 15,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}

results_ntree <- resamples(store_maxtrees)
summary(results_ntree)
# The number of trees corresponding to the highest in-sample Accuracy is 50

# Fit the model with optimized parameters:
clRf <- train(y=dat[, 17],                
              x=dat[,1:16],
              data=dat,
              method = "rf",
              metric = "Accuracy",
              tuneGrid = tuneGrid,
              trControl = trControl,
              preProcess = c('center', 'scale'),
              importance = TRUE,
              ntree = 50,
              maxnodes = 15)

# Evaluate the model:
predictionRf <-predict(clRf, test)
recapRf <- confusionMatrix(predictionRf, test$Revenue, positive = '1')
recapRf

# Get and merge metrics
Metric <- recapRf$overall[c(1,2)]
Metric <- as.data.frame(Metric)
performance <- recapRf$byClass[c(1,2,5,6,7)]
performance <- as.data.frame(performance)
performance <- performance %>% rename(Metric = performance)
stats_rf <- bind_rows(Metric, performance)

# ROC:
pred_rf <- prediction(as.numeric(predictionRf), as.numeric(test$Revenue))
roc_rf <- performance(pred_rf, "tpr", "fpr")
plot(roc_rf, col="red", lwd=3)
abline(a=0, b=1, lwd=3, lty=2)

# Area Under the Curve:
auc_rf <- performance(pred_rf, measure = "auc")

# Extract AUC
perf <- unlist(auc_rf@y.values)
perf <- as.data.frame(perf)
row.names(perf)[1] <- "AUC"
perf <- perf %>% rename(Metric=perf)

# Overall performance summary
rf_summary <- rbind(stats_rf, perf)
rf_summary <- rf_summary %>% rename('RF Performance' = Metric)
View(rf_summary)

# Show the importance of variables in terms of error reductions - higher is better
clRfImp <- varImp(clRf)
plot(clRfImp, top = 5)




####### Log Regression #########
dat$Month <- as.factor(dat$Month)
dat$VisitorType <- as.factor(dat$VisitorType)

# Train model
log_reg_model <- train(
  x = dat[, 1:16],
  y = dat[, 17],
  method = "glm",
  trControl = trControl,
  metric = "Accuracy",
  family = "binomial"
)

# Evaluate the model:
predLog <-predict(log_reg_model, test)
recapLog <- confusionMatrix(predLog, test$Revenue, positive = '1')
recapLog

# Get and merge metrics
Metric <- recapLog$overall[c(1,2)]
Metric <- as.data.frame(Metric)
performance <- recapLog$byClass[c(1,2,5,6,7)]
performance <- as.data.frame(performance)
performance <- performance %>% rename(Metric = performance)
stats_log <- bind_rows(Metric, performance)

# ROC:
pred_log <- prediction(as.numeric(predLog), as.numeric(test$Revenue))
roc_log <- performance(pred_log, "tpr", "fpr")
plot(roc_log, col="red", lwd=3)
abline(a=0, b=1, lwd=3, lty=2)

# Area Under the Curve:
auc_log <- performance(pred_log, measure = "auc")

# Extract AUC
perL <- unlist(auc_log@y.values)
perL <- as.data.frame(perL)
row.names(perL)[1] <- "AUC"
perL <- perL %>% rename(Metric=perL)

# Overall performance summary
log_summary <- rbind(stats_log, perL)
log_summary <- log_summary %>% rename('Logistic Regression Performance' = Metric)
View(log_summary)

# Show the importance of variables in terms of error reductions - higher is better
log_reg_modelImp <- varImp(log_reg_model)
plot(log_reg_modelImp, top = 5)


# Overall comparison
overall_comp <- cbind(rf_summary, log_summary)
colnames(overall_comp) <- c("Random Forest", "Logistic Regression")
View(overall_comp)
