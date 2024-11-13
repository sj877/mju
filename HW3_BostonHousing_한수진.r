
# HW3_한수진_BostonHousing.r - Part 2: Predicting Housing Median Prices

# Load necessary libraries
library(class)
library(caret)

# Load Boston Housing data (assuming it's available as a CSV in the working directory)
boston_data <- read.csv("BostonHousing.csv")

# 1. 데이터 분할: training (60%), validation (40%)
set.seed(123)  # Reproducibility
trainIndex <- createDataPartition(boston_data$MEDV, p=0.6, list=FALSE)
trainData <- boston_data[trainIndex, ]
validationData <- boston_data[-trainIndex, ]

# 2. 데이터 표준화 (Normalization)
# Only the first 12 columns are used for prediction
train_norm <- scale(trainData[, 1:12])
val_norm <- scale(validationData[, 1:12])

# Initialize variables to store RMSE for each k
rmse_values <- numeric(5)

# Run k-NN for k = 1 to 5
for (k in 1:5) {
  # Perform k-NN using class::knn() function
  knn_predictions <- class::knn(train=train_norm, test=val_norm, cl=trainData$MEDV, k=k)
  
  # Calculate RMSE for validation set
  rmse_values[k] <- sqrt(mean((as.numeric(knn_predictions) - validationData$MEDV)^2))
}

# Display the RMSE values for each k
cat("Validation RMSE for k=1 to k=5:", rmse_values, "\n")

# 3. 최적의 k로 새로운 데이터 예측
# 새로운 데이터로 MEDV 예측
new_data <- data.frame(CRIM=0.2, ZN=0, INDUS=7, CHAS=0, NOX=0.538, RM=6, AGE=62, DIS=4.7, RAD=4, TAX=307, PTRATIO=21, LSTAT=10)
scaled_new_data <- scale(new_data, center=attr(train_norm, "scaled:center"), scale=attr(train_norm, "scaled:scale"))
best_k <- which.min(rmse_values)
knn_pred_new <- class::knn(train=train_norm, test=scaled_new_data, cl=trainData$MEDV, k=best_k)

cat("Best k:", best_k, "\n")
cat("Prediction for new area:", as.numeric(knn_pred_new), "\n")
