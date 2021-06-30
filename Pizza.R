# Pizza Values 

# Regression, numeric inputs

# Dataset Description: https://www.kaggle.com/shishir349/can-pizza-be-healthy


# load libraries
library(mlbench)
library(caret)
library(corrplot)

# attach the pizza dataset
pizza = read.csv("C:/Users/mihby/Desktop/MLPizza/Pizza.csv")

pizza <- na.omit(pizza)
anyNA(pizza)

pizza[,1] <- as.numeric(factor(pizza[,1]))
pizza[,2] <- as.numeric(pizza[,2])

# normalize <- function(x) {
#   return ((x - min(x)) / (max(x) - min(x)))
# }
# for (i in 1:9){
#   pizza[,i]<-normalize(pizza[,i])
# }


# Split out validation dataset
# create a list of 90% of the rows in the original dataset we can use for training
set.seed(7)
validation_index <- createDataPartition(pizza$cal, p=0.90, list=FALSE)
# select 10% of the data for validation
validation <- pizza[-validation_index,]
# use the remaining 90% of data to training and testing the models
datasetTrain <- pizza[validation_index,]


# Summarize data

# dimensions of dataset
dim(datasetTrain)

# list types for each attribute
sapply(datasetTrain, class)


head(datasetTrain, n=20)

# summarize attribute distributions
summary(datasetTrain)


# summarize correlations between input variables
cor(datasetTrain[,1:9])


# Univaraite Visualization

# histograms each attribute
par(mfrow=c(2,5))
for(i in 1:9) {
  hist(pizza[,i], main=names(pizza)[i])
}

# density plot for each attribute
par(mfrow=c(2,5))
for(i in 1:9) {
  plot(density(pizza[,i]), main=names(pizza)[i])
}

# boxplots for each attribute
par(mfrow=c(2,5))
for(i in 1:9) {
  boxplot(pizza[,i], main=names(pizza)[i])
}

# boxplot(pizza$fat)$out
# outliers <- boxplot(pizza$fat, plot=FALSE)$out
# pizza <- pizza[-which(pizza$fat %in% outliers),]
# 
# boxplot(pizza$prot)$out
# outliers <- boxplot(pizza$prot, plot=FALSE)$out
# pizza <- pizza[-which(pizza$prot %in% outliers),]
# 
# boxplot(pizza$prot)$out
# outliers <- boxplot(pizza$prot, plot=FALSE)$out
# pizza <- pizza[-which(pizza$prot %in% outliers),]
# 
# boxplot(pizza$fat)$out
# outliers <- boxplot(pizza$fat, plot=FALSE)$out
# pizza <- pizza[-which(pizza$fat %in% outliers),]

# dim(pizza)

# Multivariate Visualizations

# scatterplot matrix
pairs(pizza[,1:8])

# correlation plot
correlations <- cor(pizza[,1:8])
corrplot(correlations, method="circle")


# Evaluate Algorithms

# Run algorithms using 10-fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# lm
set.seed(7)
fit.lm <- train(cal~., data=datasetTrain, method="lm", metric=metric, preProc=c("center", "scale"), trControl=control)
# GLM
set.seed(7)
fit.glm <- train(cal~., data=datasetTrain, method="glm", metric=metric, preProc=c("center", "scale"), trControl=control)
# GLMNET
set.seed(7)
fit.glmnet <- train(cal~., data=datasetTrain, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=control)
# SVM
set.seed(7)
fit.svm <- train(cal~., data=datasetTrain, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control)
# CART
set.seed(7)
fit.cart <- train(cal~., data=datasetTrain, method="rpart", metric=metric, preProc=c("center", "scale"), trControl=control)
# kNN
set.seed(7)
fit.knn <- train(cal~., data=datasetTrain, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
# Compare algorithms
results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm, CART=fit.cart, KNN=fit.knn))
summary(results)
dotplot(results)


# Evaluate Algorithms: with Feature Selection step

# remove correlated attributes
# find attributes that are highly corrected
set.seed(7)
cutoff <- 0.70
correlations <- cor(datasetTrain[,1:8])
highlyCorrelated <- findCorrelation(correlations, cutoff=cutoff)
for (value in highlyCorrelated) {
  print(names(datasetTrain)[value])
}
# create a new dataset without highly corrected features
dataset_features <- datasetTrain[,-highlyCorrelated]
dim(dataset_features)

# Run algorithms using 10-fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# lm
set.seed(7)
fit.lm <- train(cal~., data=dataset_features, method="lm", metric=metric, preProc=c("center", "scale"), trControl=control)
# GLM
set.seed(7)
fit.glm <- train(cal~., data=dataset_features, method="glm", metric=metric, preProc=c("center", "scale"), trControl=control)
# GLMNET
set.seed(7)
fit.glmnet <- train(cal~., data=dataset_features, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=control)
# SVM
set.seed(7)
fit.svm <- train(cal~., data=dataset_features, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control)
# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(cal~., data=dataset_features, method="rpart", metric=metric, tuneGrid=grid, preProc=c("center", "scale"), trControl=control)
# kNN
set.seed(7)
fit.knn <- train(cal~., data=dataset_features, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
# Compare algorithms
feature_results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm, CART=fit.cart, KNN=fit.knn))
summary(feature_results)
dotplot(feature_results)


# Evaluate Algorithnms: with Box-Cox Transform

# Run algorithms using 10-fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# lm
set.seed(7)
fit.lm <- train(cal~., data=datasetTrain, method="lm", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# GLM
set.seed(7)
fit.glm <- train(cal~., data=datasetTrain, method="glm", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# GLMNET
set.seed(7)
fit.glmnet <- train(cal~., data=datasetTrain, method="glmnet", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# SVM
set.seed(7)
fit.svm <- train(cal~., data=datasetTrain, method="svmRadial", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(cal~., data=datasetTrain, method="rpart", metric=metric, tuneGrid=grid, preProc=c("center", "scale", "BoxCox"), trControl=control)
# kNN
set.seed(7)
fit.knn <- train(cal~., data=datasetTrain, method="knn", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# Compare algorithms
transform_results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm, CART=fit.cart, KNN=fit.knn))
summary(transform_results)
dotplot(transform_results)


# Ensemble Methods
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# Random Forest
set.seed(7)
fit.rf <- train(cal~., data=datasetTrain, method="rf", metric=metric, preProc=c("BoxCox"), trControl=control)
# Stochastic Gradient Boosting
set.seed(7)
fit.gbm <- train(cal~., data=datasetTrain, method="gbm", metric=metric, preProc=c("BoxCox"), trControl=control, verbose=FALSE)
# Cubist
set.seed(7)
fit.cubist <- train(cal~., data=datasetTrain, method="cubist", metric=metric, preProc=c("BoxCox"), trControl=control)
# Compare algorithms
ensemble_results <- resamples(list(RF=fit.rf, GBM=fit.gbm, CUBIST=fit.cubist))
summary(ensemble_results)
dotplot(ensemble_results)


# look at parameters used for Cubist--best model
print(fit.cubist)


x <- validation[,1:8]
y <- validation[,9]


predictions <- predict(fit.cubist, newdata=x)
print(predictions)

# calculate RMSE
rmse <- RMSE(predictions, y)
r2 <- R2(predictions, y)
print(rmse)

# save the model to disk
saveRDS(fit.cubist, "MyFinalModel2.rds")
#############################################

#use the model for prediction
print("load the model")
model <- readRDS("MyFinalModel2.rds")

# make a predictions on "new data" using the final model
finalPredictions <- predict(model, x)
print(finalPredictions)
rmse <- RMSE(finalPredictions, y)
print(rmse)
#for classification problem only
#confusionMatrix(finalpredictions, y)




