# Load required libraries
library(tidyverse)
library(xgboost)
library(caret)
library(recipes)
library(MLmetrics)

# Read the data
homes_data <- read.csv('homes_data_final.csv')

# Select features
selected_features <- c('bathrooms', 'bedrooms', 'lotSize', 'median_income', 'property_age',
                      'altitude', 'homeType', 'price', 'coffee_shop_count', 'distance_to_coast',
                      'crime_index')

df_clean <- homes_data[, selected_features]

# Define continuous variables
continuous_vars <- c('bathrooms', 'bedrooms', 'lotSize', 'median_income', 'property_age',
                    'altitude', 'coffee_shop_count', 'distance_to_coast', 'crime_index')

# Clean numeric columns: remove non-numeric characters and convert to numeric
for(col in continuous_vars) {
  df_clean[[col]] <- as.numeric(gsub("[^0-9.]", "", df_clean[[col]]))
}

# Separate features and target
X <- df_clean[, !names(df_clean) %in% c('price')]
y <- df_clean$price

# Print data types and check for NAs
print("Feature data types:")
print(sapply(X, class))
print(paste('Total NAs in X:', sum(is.na(X))))
print(paste('Total NAs in y:', sum(is.na(y))))

# Remove rows with NAs
valid_indices <- complete.cases(X) & !is.na(y)
X <- X[valid_indices, ]
y <- y[valid_indices]

# Create fold indices
set.seed(42)
folds <- createFolds(y, k = 5, list = TRUE, returnTrain = FALSE)

# Initialize storage for results
fold_results_xgb <- list()
residuals_list <- numeric()
y_test_list <- numeric()
y_pred_list <- numeric()

# Perform k-fold cross validation
for(fold in 1:length(folds)) {
  cat(sprintf("\nTraining XGBoost on fold %d...\n", fold))
  
  # Split data into train and test
  test_index <- folds[[fold]]
  X_train <- X[-test_index, ]
  X_test <- X[test_index, ]
  y_train <- y[-test_index]
  y_test <- y[test_index]
  
  # Preprocessing
  # Create recipe for preprocessing
  preprocessing_recipe <- recipe(~ ., data = X_train) %>%
    step_dummy(homeType) %>%
    step_impute_knn(all_predictors(), neighbors = 5) %>%
    step_scale(all_numeric_predictors())
  
  # Prepare the preprocessing
  prep_recipe <- prep(preprocessing_recipe)
  
  # Apply preprocessing
  X_train_preprocessed <- bake(prep_recipe, new_data = X_train)
  X_test_preprocessed <- bake(prep_recipe, new_data = X_test)
  
  # Convert to matrix format for xgboost
  dtrain <- xgb.DMatrix(as.matrix(X_train_preprocessed), label = y_train)
  dtest <- xgb.DMatrix(as.matrix(X_test_preprocessed), label = y_test)
  
  # Set XGBoost parameters
  params <- list(
    objective = "reg:squarederror",
    colsample_bytree = 0.3,
    learning_rate = 0.1,
    max_depth = 5,
    alpha = 10
  )
  
  # Train XGBoost model
  xgb_model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 500
  )
  
  # Make predictions
  y_pred <- predict(xgb_model, dtest)
  
  # Calculate metrics
  rmse <- sqrt(mean((y_test - y_pred)^2))
  mae <- mean(abs(y_test - y_pred))
  r2 <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)
  
  # Store results
  fold_results_xgb[[fold]] <- list(
    MAE = mae,
    RMSE = rmse,
    'R-squared' = r2
  )
  
  cat(sprintf("Fold %d results: MAE=%.4f, RMSE=%.4f, R-squared=%.4f\n", 
              fold, mae, rmse, r2))
  
  # Store residuals and predictions
  residuals_list <- c(residuals_list, y_test - y_pred)
  y_test_list <- c(y_test_list, y_test)
  y_pred_list <- c(y_pred_list, y_pred)
}

# Calculate average metrics across all folds
if(length(fold_results_xgb) > 0) {
  average_results_xgb <- data.frame(
    MAE = mean(sapply(fold_results_xgb, function(x) x$MAE)),
    RMSE = mean(sapply(fold_results_xgb, function(x) x$RMSE)),
    R_squared = mean(sapply(fold_results_xgb, function(x) x$`R-squared`))
  )
  
  cat('\nXGBoost Average results across all folds:\n')
  print(average_results_xgb)
} else {
  cat("No valid fold results to calculate averages.\n")
}


#EXPORT MODEL AND PREPROCESSING

# Export the trained model (using the last fold's model as an example)
saveRDS(xgb_model, "xgboost_model.rds")

# Export the preprocessing recipe (contains both scaling and encoding information)
saveRDS(prep_recipe, "preprocessing_recipe.rds")
