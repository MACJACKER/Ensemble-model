## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(simpleEnsembleGroup8)

## -----------------------------------------------------------------------------
data(iris)
test_model <- fit_linear_model(iris$Sepal.Length, iris[, 3:4], add_intercept = TRUE, bagging = FALSE)
print(test_model$summary)

#Comparing with r builtin results
lm_model <- lm(Sepal.Length ~ Petal.Length + Petal.Width, data = iris)
summary(lm_model)

## ----cars---------------------------------------------------------------------
# Load the iris dataset
data(iris)
# Convert species to a numeric factor for regression; let's predict Sepal.Length
iris$Species <- as.numeric(as.factor(iris$Species))
# Splitting the dataset into training and testing sets (70:30)
set.seed(123)  # for reproducibility
sample_size <- floor(0.7 * nrow(iris))
train_indices <- sample(seq_len(nrow(iris)), size = sample_size)
train <- iris[train_indices, ]
test <- iris[-train_indices, ]
# Selecting predictors and the response variable
train_X <- train[, c("Sepal.Width", "Petal.Length", "Petal.Width", "Species")]
train_y <- train$Sepal.Length
test_X <- test[, c("Sepal.Width", "Petal.Length", "Petal.Width", "Species")]
# Fit the model using the training data
fitted_model <- fit_linear_model(train_y, train_X, add_intercept = TRUE, bagging = FALSE)
# Make predictions on the testing set
predictions <- predict_model(fitted_model, test_X, type = "response")
# Print the model summary and predictions
print(fitted_model$summary)
print(predictions)


## -----------------------------------------------------------------------------
# Load the iris dataset if not already loaded
data(iris)

# Setting up the predictors and the target variable for Gaussian model
X_gaussian <- iris[, c("Sepal.Width", "Petal.Width", "Sepal.Length")]
y_gaussian <- iris$Petal.Length

# Splitting the dataset into training and testing sets (70:30)
set.seed(123)  # for reproducibility
sample_size <- floor(0.7 * nrow(iris))
train_indices <- sample(seq_len(nrow(iris)), size = sample_size)
train_X <- as.matrix(X_gaussian[train_indices, ])
train_y <- y_gaussian[train_indices]
test_X <- as.matrix(X_gaussian[-train_indices, ])
test_y <- y_gaussian[-train_indices]

# Fit the ridge regression model using the training data
fitted_ridge_model <- fit_ridge_model(train_y, train_X, model_type = "gaussian", add_intercept = TRUE, bagging = FALSE)

# Print the fitted model's details for inspection
print(fitted_ridge_model$model)  # Print the model object itself if needed
print(fitted_ridge_model$coefficients)  # Coefficients of the fitted model

# Make predictions on the testing set
test_predictions <- predict_model(fitted_ridge_model, test_X, type = "response")

# Print predictions
print(test_predictions)


## -----------------------------------------------------------------------------
# Load the iris dataset if not already loaded
data(iris)
# Setting up the predictors and the target variable for Gaussian model
X_gaussian <- iris[, c("Sepal.Width", "Petal.Width", "Sepal.Length")]
y_gaussian <- iris$Petal.Length
# Splitting the dataset into training and testing sets (70:30)
set.seed(123)  # for reproducibility
sample_size <- floor(0.7 * nrow(iris))
train_indices <- sample(seq_len(nrow(iris)), size = sample_size)
train_X <- as.matrix(X_gaussian[train_indices, ])
train_y <- y_gaussian[train_indices]
test_X <- as.matrix(X_gaussian[-train_indices, ])
test_y <- y_gaussian[-train_indices]
# Fit the lasso regression model using the training data
fitted_lasso_model <- fit_lasso_model(train_y, train_X, model_type = "gaussian", add_intercept = TRUE, bagging = FALSE)
# Print the fitted model's details for inspection
print(fitted_lasso_model$model)  # Print the model object itself if needed
print(fitted_lasso_model$coefficients)  # Coefficients of the fitted model
# Make predictions on the testing set
test_predictions <- predict_model(fitted_lasso_model, test_X, type = "response")
# Print predictions
print(test_predictions)

## -----------------------------------------------------------------------------
# Load the iris dataset
data(iris)
# Prepare Gaussian model data
X_gaussian <- as.matrix(iris[, c("Sepal.Width", "Petal.Width", "Sepal.Length")])
y_gaussian <- iris$Petal.Length
# Splitting the dataset into training and testing sets (70:30)
set.seed(123)  # for reproducibility
sample_size <- floor(0.7 * nrow(iris))
train_indices <- sample(seq_len(nrow(iris)), size = sample_size)
train_X <- as.matrix(X_gaussian[train_indices, ])
train_y <- y_gaussian[train_indices]
test_X <- as.matrix(X_gaussian[-train_indices, ])
test_y <- y_gaussian[-train_indices]

# Fit Elastic Net for Gaussian model without bagging
result_elastic_net_gaussian <- fit_elastic_net_model(train_y, train_X, alpha = 0.5, model_type = "gaussian", add_intercept = TRUE, bagging = FALSE)
print(result_elastic_net_gaussian$model)

# Make predictions on the testing set(without bagging)
test_predictions <- predict_model(fitted_lasso_model, test_X, type = "response")
# Print predictions
print(test_predictions)


# Fit Elastic Net for Gaussian model with bagging
result_elastic_net_gaussian_bagging <- fit_elastic_net_model(train_y, train_X, alpha = 0.5, model_type = "gaussian", add_intercept = TRUE, bagging = TRUE, R = 50)
print(result_elastic_net_gaussian_bagging)

# Make predictions on the testing set(with bagging)
test_predictions <- predict_model(fitted_lasso_model, test_X, type = "response")
# Print predictions
print(test_predictions) 

## -----------------------------------------------------------------------------
# Load the iris dataset
data(iris)
# Prepare Gaussian model data
X_gaussian <- as.matrix(iris[, c("Sepal.Width", "Petal.Width", "Sepal.Length")])
y_gaussian <- iris$Petal.Length
# Splitting the dataset into training and testing sets (70:30)
set.seed(123)  # for reproducibility
sample_size <- floor(0.7 * nrow(iris))
train_indices <- sample(seq_len(nrow(iris)), size = sample_size)
train_X <- as.matrix(X_gaussian[train_indices, ])
train_y <- y_gaussian[train_indices]
test_X <- as.matrix(X_gaussian[-train_indices, ])
test_y <- y_gaussian[-train_indices]

# Fit Random Forest for Gaussian model (Regression)
result_rf_gaussian <- fit_random_forest_model(train_y, train_X, model_type = 'gaussian')
print(result_rf_gaussian$model)

# Make predictions on the testing set
test_predictions <- predict_model(result_rf_gaussian, test_X, type = "response")
# Print predictions
print(test_predictions)


## -----------------------------------------------------------------------------
# Load the iris dataset 
data(iris)

# Setting up the data
X <- iris[, c("Sepal.Length", "Sepal.Width", "Petal.Width")]  # Exclude the target variable
y <- iris$Petal.Length  # We will predict the Petal.Length

# Define the list of model fitting functions
model_list <- c("fit_elastic_net_model", "fit_ridge_model", "fit_lasso_model")

# Fit the ensemble model using the training data
ensemble_results <- ensemble_model_fitting(y, X, model_type = 'gaussian', model_list = model_list)

# Print ensemble results and individual model details
print(ensemble_results$combined_predictions)
# print(ensemble_results$model_details) # you can uncomment this line of code and run it

## -----------------------------------------------------------------------------
# 1. **Load the dataset**: First, load the `mtcars` dataset into your R environment.
data(mtcars) 
X <- mtcars[, c("hp", "wt")] # Predictor variables 
y <- as.numeric(mtcars$am == 1) # Response variable (binary)
y
my_model_coefficients <- fit_logistic_model(y,X)
print(my_model_coefficients$summary)

# Built-in Logistic Regression Model
my_model <- glm(y ~ hp+wt, data = mtcars, family = binomial(link = "logit"))

# Print model summary
print(summary(my_model))

## -----------------------------------------------------------------------------
#Example for Binomial model
data(mtcars)
X <- mtcars[, c("hp", "wt")]
y <- ifelse(mtcars$am == 1, 1, 0)  # Converting 'am' to a binary outcome
result_ridge_binomial <- fit_ridge_model(y, X, model_type = "binomial", add_intercept = TRUE, bagging = TRUE, R = 50)
print(result_ridge_binomial)

## -----------------------------------------------------------------------------
data(mtcars)
X <- mtcars[, c("hp", "wt")]
y <- ifelse(mtcars$am == 1, 1, 0)  # Converting 'am' to a binary outcome
result_lasso_binomial <- fit_lasso_model(y, X, model_type = "binomial", add_intercept = TRUE, bagging = TRUE, R = 50)
print(result_lasso_binomial)

## -----------------------------------------------------------------------------
# Example for Binomial model
data(mtcars)
X <- mtcars[, c("hp", "wt")]
y <- ifelse(mtcars$am == 1, 1, 0)  # Converting 'am' to a binary outcome
result_elastic_net_binomial <- fit_elastic_net_model(y, X, alpha = 0.5, model_type = "binomial", add_intercept = TRUE, bagging = TRUE, R = 50)
print(result_elastic_net_binomial)

## -----------------------------------------------------------------------------
data(mtcars)
X <- mtcars[, c("hp", "wt")]
y <- as.numeric(mtcars$am != 0)
result_rf_binomial <- fit_random_forest_model(y, X, model_type = "binomial")
print(result_rf_binomial)

## -----------------------------------------------------------------------------
#Assuming `fit_linear_model` and `fit_random_forest_model` are used.
data(mtcars)
X <- mtcars[, -which(names(mtcars) == "mpg")]
y <- mtcars$am # For gaussian; use a binary response for 'binomial'

model_list <- c("fit_logistic_model", "fit_random_forest_model","fit_lasso_model")

results <- ensemble_model_fitting(y, X, model_type = 'binomial', model_list = model_list)
print(results$combined_predictions)

# To view individual model details and predictions:
# print(results$model_details) # you can uncomment this line of code and run it

## -----------------------------------------------------------------------------
X <- mtcars[, c("hp", "wt")]
y <- as.numeric(mtcars$am == 1)  # Response variable (binary)
num_rows <- nrow(mtcars)
num_train <- round(0.7 * num_rows)
train_indices <- sample(1:num_rows, num_train)
training_data <- X[train_indices, ]
training_labels <- y[train_indices]
testing_data <- X[-train_indices, ]
testing_labels <- y[-train_indices]
newdata <- testing_data
model<- fit_logistic_model(y,X)
predict_model(model, newdata, type = "response", threshold = 0.5)

## -----------------------------------------------------------------------------
# Set the seed for reproducibility
set.seed(123)

# Define the number of observations and predictors
num_observations = 50
num_predictors = 100

# Generate a random dataset
# Each row corresponds to an observation
# Each column corresponds to a predictor
X <- matrix(rnorm(num_observations * num_predictors), 
               nrow = num_observations, 
               ncol = num_predictors)

# Convert matrix to data frame
X <- as.data.frame(X)

# Label the features
feature_labels <- paste("Feature", 1:num_predictors, sep="")
names(X) <- feature_labels

y <- rnorm(num_observations)

## -----------------------------------------------------------------------------
# Define the number of top predictors to select
K <- 10
# Call the select_informative_predictors function
result <- select_informative_predictors(X,K)
result$top_predictors_data
# Print the top K predictors and their scores
cat("Top Predictors:\n")
print(result$predictor_names)
cat("\nScores:\n")
print(result$scores)

## -----------------------------------------------------------------------------
model_list <- c("fit_linear_model", "fit_ridge_model", "fit_random_forest_model")
ensemble_fit <- ensemble_model_fitting(y, result$top_predictors_data, model_type = "gaussian", model_list)
print(ensemble_fit$combined_predictions)

# To view individual model details and predictions:
# print(results$model_details) # you can uncomment this line of code and run it

