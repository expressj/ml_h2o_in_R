# install.packages("h2o")
# file.copy(from = "simple_using_h2o.Rmd", to = "play_with_h2o.Rmd")

require(h2o)
h2o.init(nthreads = 3, max_mem_size = "6g")

# 데이터 준비
set.seed(1234)
train_idx <- sample(1:nrow(iris), size = 0.7 * nrow(iris), replace = FALSE)
train_iris <- iris[train_idx, ]
test_iris <- iris[-train_idx, ]

# 데이터를 h2o 상에 올려두기
train_iris_h2o <- as.h2o(train_iris, "train_iris_h2o")
test_iris_h2o <- as.h2o(test_iris, "test_iris_h2o")

# Cross Validation with H2O ----------------
rf_model_cv <- h2o.randomForest(
  x = features,
  y = target,
  training_frame = train_iris_h2o,
  model_id = "rf_model",
  ntrees = 100, 
  nfolds = 5, # 5-fold cross validation
  keep_cross_validation_predictions = TRUE, # keep cross validation predictions, TRUE로 설정해두면 Ensemble에 이용할 수 있다.
  fold_assignment = "Modulo"# fold assignment 방식 설정, Modulo로 설정해두면 Ensemble에 이용할 수 있다.
)

summary(rf_model_cv)

# Grid Search for hyperparameter with H2O ----------------

hyper_params <- list(
  ntrees = c(10, 50, 100, 150, 200),
  max_depth = c(10, 20, 30),
  min_rows = c(1, 3, 5, 10),
  nbins = c(15, 20, 25)
) # Hyperparameter 값 지정

rf_grid <- h2o.grid(algorithm = "randomForest",
  x = features,
  y = target,
  training_frame = train_iris_h2o,
  grid_id = "rf_grid", 
  hyper_params = hyper_params
)
# 위에서 지정한 Hyperparameter set을 이용해 모든 조합에 대해서 Grid Search 수행

# 원하는 metric을 기준으로 모델들을 나열 할 수 있음.
rf_grid_sorted <- h2o.getGrid("rf_grid", sort_by = "mean_per_class_accuracy", decreasing = TRUE)
best_model <- h2o.getModel(rf_grid_sorted@model_ids[[1]]) # 가장 첫번째 모델이 설정한 metric 기준 best model!

summary(best_model)

# AutoML 수행 -----------
automl <- h2o.automl(
  x = features,
  y = target,
  training_frame = train_iris_h2o,
  max_models = 100,
  sort_metric = "logloss"
)

pred_iris_automl <- as.data.frame(h2o.predict(automl, newdata = test_iris_h2o))
test_iris$pred_automl <- pred_iris_automl$predict

with(test_iris, table(Species, pred_automl, dnn = c("Real", "Predict")))

