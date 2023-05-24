#code is from https://www.kaggle.com/code/burakdilber/heart-failure-eda-preprocessing-and-10-models

# Necessary installations
#install.packages("tidyverse")
#install.packages('GGally')
#install.packages('superml')
#install.packages('caret')
#install.packages('Boruta')
#install.packages('tidymodels')
#install.packages('baguette')
#install.packages('discrim')
#install.packages('bonsai')
#install.packages('xgboost')
#install.packages('kknn')
#install.packages('shapr')
#install.packages('vip')

# Necessary imports
library(tidyverse)
library(GGally)
library(superml)
library(caret)
library(Boruta)
library(tidymodels)
library(baguette)
library(discrim)
library(bonsai)
library(xgboost)
library(kknn)
library(shapr)
library(vip)

################################################################################

# Read in the data
heart <- read_csv("data/heart.csv")
heart %>% glimpse()
heart %>% head()
heart %>% summary()

# Transform character to factor
heart <- heart %>%
  mutate_if(is.character, as.factor)
heart %>% summary()
heart %>% head()

# Check for missing values
heart %>%
  select(everything()) %>%
summarise_all(funs(sum(is.na(.))))
heart %>% head()

# Label encoding
lbl = LabelEncoder$new()
heart$Sex <- lbl$fit_transform(heart$Sex)
heart$ST_Slope = lbl$fit_transform(heart$ST_Slope)
heart %>% summary()
heart %>% head()

# One hot encoding
dummy_data <- dummyVars(" ~ .", data = heart)
heart <- as.data.frame(predict(dummy_data, heart))
heart %>% summary()
heart %>% head()

# Feature selection
feature_select <- Boruta(HeartDisease ~ ., data = heart)
feature_select$finalDecision
heart <- heart %>%
  select(-c("RestingECG.Normal", "RestingECG.ST"))
glimpse(heart)
heart %>% head()

# Convert factor to dependent variable
heart$HeartDisease <- as.factor(heart$HeartDisease)
glimpse(heart)
heart %>% head()

# Split data
model_recipe <- 
  recipe(HeartDisease ~ ., data = heart)
set.seed(123)
heart_split <- initial_split(heart, prop = 0.80)
heart_split
heart_train <- training(heart_split)
heart_test  <- testing(heart_split)
heart_cv <- vfold_cv(heart_train, v = 10)

################################################################################

# Random Forest Algorithm, best result given accuracy
rf_model <- 
  rand_forest(mode = "classification",
              mtry = tune(),
              trees = tune(),
              min_n = tune()
  ) %>%
  set_engine("ranger", importance = "impurity")
set.seed(123)
rf_wf <-
  workflow() %>%
  add_model(rf_model) %>% 
  add_recipe(model_recipe)
rf_results <-
  rf_wf %>% 
  tune_grid(resamples = heart_cv,
            metrics = metric_set(accuracy)
  )
rf_results %>%
  collect_metrics()
param_final_rf <- rf_results %>%
  select_best(metric = "accuracy")
param_final_rf
rf_wf <- rf_wf %>%
  finalize_workflow(param_final_rf)
rf_wf
rf_fit <- rf_wf %>%
  last_fit(heart_split)
rf_fit
test_performance_rf <- rf_fit %>% collect_predictions()
test_performance_rf
heart_metrics <- metric_set(accuracy, f_meas, precision, recall)
heart_metrics(data = test_performance_rf, truth = HeartDisease, estimate = .pred_class)
conf_mat(test_performance_rf, HeartDisease, .pred_class)

# Extract the random forest model
rf_wf <- rf_wf %>%
  fit(heart_train)
rf_wf
model_rf = extract_fit_parsnip(rf_wf)

# Try some predictions
predict(model_rf, heart_test[0:5,1:15], type='class')
predict(model_rf, heart_test[0:5,1:15], type='prob')

# Save feature importances of the random forest model
feature_importance_rf = vip(model_rf)
#saveRDS(feature_importance_rf$data, file = "explain/feature_importance_rf.rds")

# Save the random forest model
#saveRDS(model_rf, file = "model/rf_heart.rds") # niet meer nodig

################################################################################

# Extreme Gradient Boosting, second best model given accuracy
xgboost_model <- 
  boost_tree( mode = "classification",
              mtry = tune(),
              trees = tune(),
              min_n = tune(),
              tree_depth = tune(),
              learn_rate = tune(),
              loss_reduction = tune(),
              sample_size = tune(),
              stop_iter = tune()
  ) %>%
  set_engine("xgboost", importance = "impurity")

set.seed(123)
xgboost_wf <-
  workflow() %>%
  add_model(xgboost_model) %>% 
  add_recipe(model_recipe)
xgboost_wf
xgboost_results <-
  xgboost_wf %>% 
  tune_grid(resamples = heart_cv,
            metrics = metric_set(accuracy)
  )
xgboost_results %>%
  collect_metrics()
param_final_xgb <- xgboost_results %>%
  select_best(metric = "accuracy")
param_final_xgb
xgboost_wf <- xgboost_wf %>%
  finalize_workflow(param_final_xgb)
xgboost_wf
xgboost_fit <- xgboost_wf %>%
  last_fit(heart_split)
test_performance_xgb <- xgboost_fit %>% collect_predictions()
test_performance_xgb
heart_metrics <- metric_set(accuracy, f_meas, precision, recall)
heart_metrics(data = test_performance_xgb, truth = HeartDisease, estimate = .pred_class)
conf_mat(test_performance_xgb, HeartDisease, .pred_class)

#extract the extreme gradient boosting model
xgboost_wf <- xgboost_wf %>%
  fit(heart_train)
xgboost_wf
model_xgb = extract_fit_parsnip(xgboost_wf)

# Try some predictions
predict(model_xgb, heart_test[0:5,1:15], type='class')
predict(model_xgb, heart_test[0:5,1:15], type='prob')

# Save feature importances of the extreme gradient boosting model
feature_importance_xgb = vip(model_xgb)
#saveRDS(feature_importance_xgb$data, file = "explain/feature_importance_xgb.rds")

# Save the extreme gradient boosting model
#saveRDS(model_xgb, file = "model/xgb_heart.rds")

################################################################################

# Ensembles of Mars Models, shared third place given accuracy
bag_mars_model <- 
  bag_mars( mode = "classification",
            num_terms = tune(),
            prod_degree = tune(),
            prune_method = tune(), 
            engine = "earth"
  ) 

set.seed(123)
bag_mars_wf <-
  workflow() %>%
  add_model(bag_mars_model) %>% 
  add_recipe(model_recipe)
bag_mars_wf

bag_mars_results <-
  bag_mars_wf %>% 
  tune_grid(resamples = heart_cv,
            metrics = metric_set(accuracy)
  )

bag_mars_results %>%
  collect_metrics()

param_final_bag_mars <- bag_mars_results %>%
  select_best(metric = "accuracy")
param_final_bag_mars

bag_mars_wf <- bag_mars_wf %>%
  finalize_workflow(param_final_bag_mars)
bag_mars_wf

bag_mars_fit <- bag_mars_wf %>%
  last_fit(heart_split)

test_performance_bag_mars <- bag_mars_fit %>% collect_predictions()
test_performance_bag_mars

heart_metrics <- metric_set(accuracy, f_meas, precision, recall)
heart_metrics(data = test_performance_bag_mars, truth = HeartDisease, estimate = .pred_class)

conf_mat(test_performance_bag_mars, HeartDisease, .pred_class)

#extract the Ensembles of Mars Models
bag_mars_wf <- bag_mars_wf %>%
  fit(heart_train)
bag_mars_wf
model_bag_mars = extract_fit_parsnip(bag_mars_wf)

# Try some predictions
predict(model_bag_mars, heart_test[0:5,1:15], type='class')
predict(model_bag_mars, heart_test[0:5,1:15], type='prob')

# Save feature importances of the Ensembles of Mars Models
feature_importance_bag_mars = vip(model_bag_mars) # does not work
#saveRDS(feature_importance_bag_mars$data, file = "explain/feature_importance_bag_mars.rds")

# Save the Ensembles of Mars Models, shared third place given accuracy
#saveRDS(model_bag_mars, file = "model/bag_mars_heart.rds")

################################################################################

#Multivariate Adaptive Regression Splines
mars_model <- 
  mars(mode = "classification",
       num_terms = tune(),
       prod_degree = tune(),
       prune_method = tune(), 
       engine = "earth"
  )

set.seed(123)
mars_wf <-
  workflow() %>%
  add_model(mars_model) %>% 
  add_recipe(model_recipe)
mars_wf

mars_results <-
  mars_wf %>% 
  tune_grid(resamples = heart_cv,
            metrics = metric_set(accuracy)
  )

mars_results %>%
  collect_metrics()

param_final_mars <- mars_results %>%
  select_best(metric = "accuracy")
param_final_mars

mars_wf <- mars_wf %>%
  finalize_workflow(param_final_mars)
mars_wf

mars_fit <- mars_wf %>%
  last_fit(heart_split)

test_performance_mars <- mars_fit %>% collect_predictions()
test_performance_mars

heart_metrics <- metric_set(accuracy, f_meas, precision, recall)
heart_metrics(data = test_performance_mars, truth = HeartDisease, estimate = .pred_class)

conf_mat(test_performance_mars, HeartDisease, .pred_class)

#extract the Multivariate Adaptive Regression Splines
mars_wf <- mars_wf %>%
  fit(heart_train)
mars_wf
model_mars = extract_fit_parsnip(mars_wf)

# Try some predictions
predict(model_mars, heart_test[0:5,1:15], type='class')
predict(model_mars, heart_test[0:5,1:15], type='prob')

# Save feature importances of the Multivariate Adaptive Regression Splines
feature_importance_mars = vip(model_mars) # does not work
#saveRDS(feature_importance_mars$data, file = "explain/feature_importance_mars.rds")

# Save the Multivariate Adaptive Regression Splines, shared third place given accuracy
#saveRDS(model_mars, file = "model/mars_heart.rds")

################################################################################

#K - Nearest Neighbor
knn_model <- 
  nearest_neighbor( mode = "classification",
                    neighbors = tune(),
                    weight_func = tune(),
                    dist_power = tune()
  ) %>%
  set_engine("kknn", importance = "impurity")

set.seed(123)
knn_wf <-
  workflow() %>%
  add_model(knn_model) %>% 
  add_recipe(model_recipe)
knn_wf

knn_results <-
  knn_wf %>% 
  tune_grid(resamples = heart_cv,
            metrics = metric_set(accuracy)
  )

knn_results %>%
  collect_metrics()

param_final_knn <- knn_results %>%
  select_best(metric = "accuracy")
param_final_knn

knn_wf <- knn_wf %>%
  finalize_workflow(param_final_knn)
knn_wf

knn_fit <- knn_wf %>%
  last_fit(heart_split)

test_performance_knn <- knn_fit %>% collect_predictions()
test_performance_knn

heart_metrics <- metric_set(accuracy, f_meas, precision, recall)
heart_metrics(data = test_performance_knn, truth = HeartDisease, estimate = .pred_class)

conf_mat(test_performance_knn, HeartDisease, .pred_class)

#extract the K - Nearest Neighbor
knn_wf <- knn_wf %>%
  fit(heart_train)
knn_wf
model_knn = extract_fit_parsnip(knn_wf)

# Try some predictions
predict(model_knn, heart_test[0:5,1:15], type='class')
predict(model_knn, heart_test[0:5,1:15], type='class')

# Save feature importances of the K - Nearest Neighbor
feature_importance_knn = vip(model_knn) # does not work
#saveRDS(feature_importance_knn$data, file = "explain/feature_importance_knn.rds")

# Save the K - Nearest Neighbor
saveRDS(model_knn, file = "model/knn_heart.rds")