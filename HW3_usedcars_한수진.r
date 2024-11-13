
# HW3_한수진_ToyotaCorolla.r

# --- Part 1: 중고차 가격 예측 (Predicting Prices of Used Cars) ---

# 필요한 라이브러리 불러오기
library(caret)

# 1. 데이터 로드 및 나누기 (training, validation, test)
# Toyota Corolla 데이터 로드
data <- read.csv("ToyotaCorolla.csv")
set.seed(123)  # 무작위성을 위한 시드 설정

# 50%의 데이터를 training set으로 분리
trainIndex <- createDataPartition(data$Price, p=0.5, list=FALSE)
trainData <- data[trainIndex, ]
remainingData <- data[-trainIndex, ]

# 남은 데이터에서 60%는 validation set으로, 40%는 test set으로 나누기
valIndex <- createDataPartition(remainingData$Price, p=0.6, list=FALSE)
validationData <- remainingData[valIndex, ]
testData <- remainingData[-valIndex, ]

# 2. 다중 선형 회귀 모델 작성
# Price를 종속 변수로, Age_08_04, KM, Fuel_Type, HP, Automatic 등 다양한 변수를 독립 변수로 설정
model <- lm(Price ~ Age_08_04 + KM + Fuel_Type + HP + Automatic + Doors +
            Quarterly_Tax + Mfg_Guarantee + Guarantee_Period + Airco +
            Automatic_Airco + CD_Player + Powered_Windows + Sport_Model + Tow_Bar, data=trainData)

# 모델의 요약 정보를 확인하여 주요 변수를 파악
summary(model)  # 유의한 변수들(주요 변수들)을 찾기 위한 요약 결과 확인

# 설명: 주요 변수는 유의 수준(p-value)에 따라 결정됩니다. 모델 요약에서 p-value가 가장 낮은 3~4개의 변수를 선택하여
# 가격 예측에 중요한 속성으로 설명하겠습니다.

# 3. 성능 평가
# RMSE와 R-squared를 이용하여 validation set과 test set에 대한 모델 성능 평가
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Validation set 예측 및 성능 평가
valPredictions <- predict(model, validationData)
valRMSE <- rmse(validationData$Price, valPredictions)
valR2 <- cor(validationData$Price, valPredictions)^2

# Test set 예측 및 성능 평가
testPredictions <- predict(model, testData)
testRMSE <- rmse(testData$Price, testPredictions)
testR2 <- cor(testData$Price, testPredictions)^2

# 결과 출력
cat("Validation Set RMSE:", valRMSE, "R-squared:", valR2, "\n")
cat("Test Set RMSE:", testRMSE, "R-squared:", testR2, "\n")
