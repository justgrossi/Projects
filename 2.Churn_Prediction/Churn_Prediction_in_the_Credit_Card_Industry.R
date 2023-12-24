library(tidyverse)
library(magrittr)
library(caret)
library(smotefamily)
library(fastDummies)
library(rpart)
library(rpart.plot)
library(ROCR)
library(e1071)

# Set seed for reproducibility
set.seed(1234)
df <- read.csv("churn.csv", header = T, stringsAsFactors = T)
sum(is.na(df))
prop.table(table(df$churned))

df %>% ggplot(aes(fct_relevel(incomeCat,
                              'Unknown',
                              'Less than $40K',
                              '$40K - $60K',
                              '$60K - $80K',
                              '$80K - $120K',
                              '$120K +')))+
  geom_bar()+
  xlab('Income Level')+
  ylab('# Customers')

df %>% ggplot(aes(x=numContacts))+
  geom_bar()+
  facet_wrap(~cardCat)+
  xlab('CC Type')+
  ylab('# Contacts')

df %>% ggplot(aes(totBought, cardCat))+
  geom_boxplot()+
  coord_flip()+
  xlab('# Ancillary Products')+
  ylab('CC Type')

# Create Train vs Test Set
train_ind <- createDataPartition(df$churned, p = .8, 
                                 list = FALSE, 
                                 times = 1)
train <- df[train_ind, ]
test <- df[-train_ind, ]

## Resolve Class Imbalance
# Create dummy numerical to feed to SMOTE
train <- dummy_cols(train,
                    select_columns = c('gender',
                                       'eduLevel',
                                       'marital',
                                       'incomeCat',
                                       'cardCat'))

# Remove original features and apply SMOTE
smote_df <- train[, -c(1,3,5,6,7,8)] %>% SMOTE(train$churned, K = 5)

# Retrieve the data
smote <- smote_df$data

# Rename target label as original
smote <- smote %>% rename(churned = class)

# Check balance
prop.table(table(smote$churned))
sum(is.na(smote))

# Remove ratio features prior to rounding SMOTED observations
smoted <- smote %>% select(-c(delta12amt, delta12count, churned)) %>% round(digits = 0)

# Adding back ratio features
smoted <- cbind(smoted, smote[c('delta12amt', 'delta12count', 'churned')])


#######  Decision Trees  #########
clTree <- rpart(churned~.
                ,smoted
                ,method="class"
                ,control = rpart.control(minsplit = 5))

# Plot cross validation results:
plotcp(clTree)

# Best complexity parameter (cp):
bestcp <- clTree$cptable[which.min(clTree$cptable[,"xerror"]),"CP"]
cp_table <- clTree$cptable
cp_table <- as.data.frame(cp_table)

cp_table %>% ggplot(aes(CP, `xerror`))+geom_line()+
  xlab('Complexity Parameter')+
  ylab('Cross Validation Error')

# Visualize best CP - 0.01
bestcp

# Prune and stop the splitting in correspondence of the best cp:
prunedTree <- prune(clTree, cp = bestcp)

#Plot pruned tree:
rpart.plot(prunedTree)

# Feature importance
datfeat <- data.frame(imp = prunedTree$variable.importance)
datfeat2 <- datfeat %>% 
  tibble::rownames_to_column() %>% 
  dplyr::rename("variable" = rowname) %>% 
  dplyr::arrange(imp) %>%
  dplyr::mutate(variable = forcats::fct_inorder(variable))

# Plot first 5 most important features
ggplot2::ggplot(datfeat2[1:5,]) +
  geom_segment(aes(x = variable, y = 0, xend = variable, yend = imp), 
               size = .8, alpha = 0.7) +
  geom_point(aes(x = variable, y = imp, col = 'blue'), 
             size = 1.2, show.legend = F) +
  coord_flip() +
  theme_bw()

# Prepare test data
test <- dummy_cols(test,
                   select_columns = c('gender',
                                      'eduLevel',
                                      'marital',
                                      'incomeCat',
                                      'cardCat'))

test$churned <- as.factor(test$churned)

# Make predictions
predictionTree <- prunedTree %>% predict(test, type="class")
recapTree <- confusionMatrix(predictionTree, test$churned)
recapTree

# Check performance
Metric <- recapTree$overall[c(1,2)]
Metric <- as.data.frame(Metric)
Performance_Tree <- recapTree$byClass[c(1,2,5,6,7)]
Performance_Tree <- as.data.frame(Performance_Tree)
Performance_Tree <- Performance_Tree %>% rename(Metric = Performance_Tree)
statsTree <- bind_rows(Metric, Performance_Tree)

# Plot ROC:
predTree <- prediction(as.numeric(predictionTree), as.numeric(test$churned))
rocTree <- performance(predTree, "tpr", "fpr")
plot(rocTree, col="red", lwd=3)
abline(a=0, b=1, lwd=3, lty=2)

# Area Under the Curve:
aucTree <- performance(predTree, measure = "auc")
ROC_Tree <- unlist(aucTree@y.values)
ROC_Tree <- as.data.frame(ROC_Tree)
row.names(ROC_Tree)[1] <- "AUC"
ROC_Tree <- ROC_Tree %>% rename(Metric=ROC_Tree)

# Performance summary
Tree_summary <- rbind(statsTree, ROC_Tree)
Tree_summary <- Tree_summary %>% rename('Decision Tree Performance' = Metric)
View(Tree_summary)



#########  GBM  #########
# Set trControl
trControl <- trainControl(method="cv", number=10)

gbmGrid <- expand.grid(
  interaction.depth = c(3, 5, 7),       
  n.trees = c(100, 200, 300),           
  shrinkage = c(0.01, 0.1, 0.3),        
  n.minobsinnode = c(5, 10, 15)     
)

# Create model
clBoosting <- train(churned~.,
                    data=smoted,
                    trControl=trControl,
                    method='gbm',
                    verbose=F,
                    tuneGrid=gbmGrid
)

# Retrieve the best tuning parameters
best_params <- clBoosting$bestTune
print(best_params) # n.trees 300 interaction.depth 7 shrinkage 0.3 n.minobsinnode 15
plot(clBoosting)

# Make predictions
predBoost <- predict(clBoosting, test)
recapBoost <- confusionMatrix(predBoost, test$churned)
recapBoost

# Get performance metrics
Metric <- recapBoost$overall[c(1,2)]
Metric <- as.data.frame(Metric)

Performance_gbm <- recapBoost$byClass[c(1,2,5,6,7)]
Performance_gbm <- as.data.frame(Performance_gbm)
Performance_gbm <- Performance_gbm %>% rename(Metric = Performance_gbm)

statsGbm <- bind_rows(Metric, Performance_gbm)

#ROC:
predGbm <- prediction(as.numeric(predBoost), as.numeric(test$churned))
roc_gbm <- performance(predGbm, "tpr", "fpr")
plot(roc_gbm, col="red", lwd=3)
abline(a=0, b=1, lwd=3, lty=2)

#Area Under the Curve:
auc_gbm <- performance(predGbm, measure = "auc")
ROC_gbm <- unlist(auc_gbm@y.values)
ROC_gbm <- as.data.frame(ROC_gbm)
row.names(ROC_gbm)[1] <- "AUC"
ROC_gbm <- ROC_gbm %>% rename(Metric=ROC_gbm)

# Create and view summary
gbm_summary <- rbind(statsGbm, ROC_gbm)
gbm_summary <- gbm_summary %>% rename('Gradient Boosting Machine Performance' = Metric)
View(gbm_summary)

# Plot first 5 most important features
underlying_gbm <- clBoosting$finalModel  # Extract the GBM model from the train object

# Get feature importance
feature_importance <- summary(underlying_gbm)[1:5,1:2]

# Plot top 5 most important features
ggplot2::ggplot(feature_importance[1:5,]) +
  geom_segment(aes(x = var, y = 0, xend = var, yend = rel.inf), size = .8, alpha = 0.7) +
  geom_point(aes(x = var, y = rel.inf, col = 'blue'), size = 1.2, show.legend = F) +
  coord_flip() +
  theme_bw()

# Overall comparison
overall_comp <- cbind(Tree_summary, gbm_summary)
colnames(overall_comp) <- c("Decision Trees", "Gradient Boosting Machine")
View(overall_comp)
