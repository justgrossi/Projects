library(tidyverse)
library(magrittr)
library(caret)
library(e1071)
library(corrplot)
library(randomForest)
library(Metrics)

# Set seed for reproducibility
set.seed(123)

en <- read.csv("energy.csv", header = T)
glimpse(en)
sum(is.na(en))
en <- en %>% relocate('consumption', .after = 'dew')

# Remove features with no variability
en[, c('year', 'sec')] <- NULL

# Check correlations
correlations <- cor(en)
corrplot(correlations, method = "color", type = "upper",
         diag = FALSE, order="FPC", tl.pos = "lt", tl.cex = 0.8)

# Standardize data frame
scaled <- scale(en)

# Remove outliers
outliers <- apply(scaled, 1, function(row) any(row > 3 | row < -3))
cleaned_df <- scaled[!outliers, ]

# Get original mean and sd
original_sd <- apply(en, 2, sd)
original_mean <- colMeans(en)
reverted <- cleaned_df * original_sd
reverted <- reverted + original_mean
reverted <- as.data.frame(reverted)

# Perform PCA
pca <- prcomp(reverted[,1:29], center = TRUE, scale. = TRUE)
summary(pca)

# Eigenvalues show 14 principal components account for 95.85% of label's variability
components <- 14

# Get values and revert to original scale
standardized_pc <- as.data.frame(pca$x[, 1:components])
reverted_original_sd <- apply(reverted, 2, sd)
reverted_original_mean <- colMeans(reverted)
reverted_pc <- standardized_pc * reverted_original_sd
reverted_pc <- reverted_pc + reverted_original_mean

# Create final data frame
final <- cbind(reverted_pc, reverted[,30])
final <- final %>% rename(consumption = 'reverted[, 30]')

# Create train vs test sets
idx <- createDataPartition(final$consumption, p = .8, 
                           list = FALSE, 
                           times = 1)
trData <- final[idx,]
teData <- final[-idx,]


##########  KNN  ##########
trControl <- trainControl(method="cv", number=10)
tuneGrid_knn <- expand.grid(k=10:50)

Knn <- train(consumption ~ ., method = "knn",
             trData
             ,tuneGrid = tuneGrid_knn
             ,trControl= trControl)

# Best model
Knn
plot(Knn)

test.features = subset(teData, select=-c(consumption))
test.target = subset(teData, select=consumption)[,1]

# Make predictions
predKnn = predict(Knn, newdata = test.features)

# Evaluate model
# Best model k=40 - R.M.S.E.: 13.18 - Rsquared: 99% - M.A.E.: 3.75 - M.S.E.: 173.77
knn_ss_total <- sum((teData$consumption - mean(teData$consumption))^2)
knn_ss_residual <- sum((teData$consumption - predKnn)^2)
knn_rsquared <- 1 - (knn_ss_residual / knn_ss_total)
print(paste("R-squared:", knn_rsquared))

knn_mae_value <- mae(predKnn, teData$consumption)
knn_mse_value <- mse(predKnn, teData$consumption)
knn_rmse_value <- rmse(predKnn, teData$consumption)

# Print the computed metrics
cat("Mean Absolute Error (MAE):", knn_mae_value, "\n")
cat("Mean Squared Error (MSE):", knn_mse_value, "\n")
cat("Root Mean Squared Error (RMSE):", knn_rmse_value, "\n")

# M.S.E. baseline: 19747.32
mean((teData$consumption-mean(teData$consumption))^2)




######### RF #######
# Search best mtry parameter value
mtry_values <- c(1:14)
tuneGrid_rf <- expand.grid(mtry=mtry_values)


mtryRf <- train(y=trData[,5],
                x=trData[, 1:14],
                data=trData,
                method = "rf",
                metric = "RMSE",
                tuneGrid = tuneGrid_rf,
                trControl = trControl,
                preProcess = c('center', 'scale'),
                ntree = 100)

# Visualize and store best value of MTRY:
print(mtryRf)
plot(mtryRf)
best_mtry <- mtryRf$bestTune$mtry

tuneGrid_rf <- expand.grid(mtry = best_mtry)

store_maxtrees <- list()
for (ntree in c(100, 150, 200, 300, 500)) {
  rf_maxtrees <- train(consumption~.,
                    data=trData,
                    method='rf',
                    metric='RMSE',
                    trControl=trControl,
                    preProcess = c('center', 'scale'),
                    tuneGrid=tuneGrid_rf,
                    ntree=ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}

results_ntree <- resamples(store_maxtrees)
summary(results_ntree)

# Fit model with best parameters
rf <- train(consumption~.,
            data = trData,
            method='rf',
            metric='RMSE',
            trControl=trControl,
            preProcess=c('center', 'scale'),
            tuneGrid=tuneGrid_rf,
            ntree=100)

# Make predictions
predictions_rf <- predict(rf, newdata = teData)

# Assess model performance
# R-squared
ss_total_rf <- sum((teData$consumption - mean(teData$consumption))^2)
ss_residual_rf <- sum((teData$consumption - predictions_rf)^2)
rsquared_rf <- 1 - (ss_residual_rf / ss_total_rf)
print(paste("R-squared:", rsquared_rf))

# Calculate evaluation metrics
mae_value_rf <- mae(predictions_rf, teData$consumption)
mse_value_rf <- mse(predictions_rf, teData$consumption)
rmse_value_rf <- rmse(predictions_rf, teData$consumption)

# Print the computed metrics
cat("Mean Absolute Error (MAE):", mae_value_rf, "\n")
cat("Mean Squared Error (MSE):", mse_value_rf, "\n")
cat("Root Mean Squared Error (RMSE):", rmse_value_rf, "\n")


# KNN
# Best model k=40
# R.M.S.E.: 13.18 - Rsquared: 99% - M.A.E.: 3.75 - M.S.E.: 173.77

# RF
# Best model ntree=100 mtry=14
#R.M.S.E.: 12.28 - Rsquared: 99% - M.A.E.: 3.87 - M.S.E: 150.91