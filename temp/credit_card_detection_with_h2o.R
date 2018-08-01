# Credit Card Fraud Detection in Kaggle
# From : https://www.kaggle.com/mlg-ulb/creditcardfraud
# License : Open Database

# Package가 없으면 설치하시기 바랍니다.
pkgs <- c("h2o", "data.table", "magrittr", "PRROC", "ggplot2")
# install.packages(pkgs)
lapply(pkgs, require, character.only = TRUE)

h2o.init(nthreads = 10, max_mem_size = "64g")

kaggle_cdc_h2o <- h2o.importFile("creditcard.csv", destination_frame = "kaggle_cdc_h2o")

variables <- names(kaggle_cdc_h2o)
variables

# Target : Class
# Target의 비율 구성은 어떻게 되어있을까?
kaggle_cdc_h2o$Class %>% 
  as.vector() %>% 
  table(dnn = "Class") %>% 
  as.data.frame(.) %>% 
  ggplot(data = ., aes(x = as.factor(Class), y = Freq)) + 
  geom_bar(stat = "identity") + 
  geom_text(aes(label = paste0(round(Freq/sum(Freq), 8) * 100, "%"))) + 
  xlab("Target : Class") + ylab("빈도수")

# ==> Imbalanced data !!


# Data split with h2o
kaggle_cdc_h2o$Class <- as.factor(kaggle_cdc_h2o$Class)

summary(kaggle_cdc_h2o, exact_quantiles=TRUE)

cdc_sp_h2o <-
  h2o.splitFrame(
    kaggle_cdc_h2o,
    ratios = 0.7,
    destination_frames = c("cdc_train", "cdc_test")
  )

# h2o.getFrame("cdc_train")
# h2o.getFrame("cdc_test")

t_var <- "Time"
target <- "Class"
features <- variables[!variables %in% c(t_var, target)]

##################################
# 1. Generating Classifier model #
##################################

## 1) GLM
glm_1 <-
  h2o.glm(
    x = features,
    y = target,
    training_frame = h2o.getFrame("cdc_train"),
    model_id = "glm_1",
    family = "binomial",
    standardize = TRUE,
    compute_p_values = TRUE,
    lambda = 0, 
    # Cross Validation : nfolds, fold_assignment, keep_cross_validation_predictions, etc..
    nfolds = 5, 
    fold_assignment = "Modulo", 
    keep_cross_validation_predictions = FALSE, 
    keep_cross_validation_fold_assignment = FALSE
  )

# model summary
summary(glm_1)

# glm의 coefficients와 p-value
glm_1@model$coefficients_table

# glm의 coefficients 값만 확인하고싶다면,
h2o.coef(glm_1)

# glm의 변수중요도
h2o.varimp_plot(glm_1)

# glm의 AUC
h2o.auc(glm_1)

# Test set의 Target 값 가져오기
target_vec_test <- h2o.getFrame("cdc_test")$Class %>% as.vector()

# Test set의 예측값 계산
pred_glm <- glm_1 %>% h2o.predict(h2o.getFrame("cdc_test")) %>% as.data.table()
pred_glm[, "Class" := target_vec_test]

# Accuracy 측정
pred_glm[, mean(Class == predict)]
pred_glm[Class == 0, mean(Class == predict)]
pred_glm[Class == 1, mean(Class == predict)]

# PR ROC curve 그려보기
fg <- pred_glm[Class == 1, p1]
bg <- pred_glm[Class == 0, p1]

pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
print(pr)
plot(pr)

# Test set을 이용한 model performance 측정
h2o.performance(glm_1, newdata = h2o.getFrame("cdc_test"))

## 2) Lasso Regression
# 단, lambda를 0이 아닌 다른 값으로 셋팅하면,
# p-value를 볼수 없음.
glm_2 <-
  h2o.glm(
    x = features,
    y = target,
    training_frame = h2o.getFrame("cdc_train"),
    model_id = "glm_2",
    family = "binomial",
    standardize = TRUE,
    nfolds = 5, 
    alpha = 1,
    fold_assignment = "Modulo",
    lambda_search = TRUE
  )

h2o.auc(glm_2)

summary(glm_2)

pred_glm_2 <- glm_2 %>% h2o.predict(h2o.getFrame("cdc_test")) %>% as.data.table()
pred_glm_2[, "Class" := target_vec_test]

# Accuracy 측정
pred_glm_2[, mean(Class == predict)]
pred_glm_2[Class == 0, mean(Class == predict)]
pred_glm_2[Class == 1, mean(Class == predict)]

# PR ROC curve 그려보기
fg <- pred_glm_2[Class == 1, p1]
bg <- pred_glm_2[Class == 0, p1]

pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
print(pr)
plot(pr)

# Test set을 이용한 model performance 측정
h2o.performance(glm_2, newdata = h2o.getFrame("cdc_test"))


## 3) RandomForest
rf_1 <-
  h2o.randomForest(
    x = features,
    y = target,
    training_frame = h2o.getFrame("cdc_train"),
    model_id = "rf_1",
    ntrees = 50,
    max_depth = 20,
    mtries = -1,
    min_rows = 1,
    nfolds = 5,
    fold_assignment = "Modulo"
  )

# model summary
summary(rf_1)

# glm의 변수중요도
h2o.varimp_plot(rf_1)

# glm의 AUC
h2o.auc(rf_1)

# Test set의 Target 값 가져오기
target_vec_test <- h2o.getFrame("cdc_test")$Class %>% as.vector()

# Test set의 예측값 계산
pred_rf <- rf_1 %>% h2o.predict(h2o.getFrame("cdc_test")) %>% as.data.table()
pred_rf[, "Class" := target_vec_test]

# Accuracy 측정
pred_rf[, mean(Class == predict)]
pred_rf[Class == 0, mean(Class == predict)]
pred_rf[Class == 1, mean(Class == predict)]

# PR ROC curve 그려보기
fg <- pred_rf[Class == 1, p1]
bg <- pred_rf[Class == 0, p1]

pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
print(pr)
plot(pr)

## 3) RandomForest - increase ntrees
# ntrees : 100
rf_2_1 <-
  h2o.randomForest(
    x = features,
    y = target,
    training_frame = h2o.getFrame("cdc_train"),
    model_id = "rf_2_1",
    ntrees = 100,
    max_depth = 20,
    mtries = -1,
    min_rows = 1,
    nfolds = 5,
    fold_assignment = "Modulo"
  )

# ntrees : 150
rf_2_2 <-
  h2o.randomForest(
    x = features,
    y = target,
    training_frame = h2o.getFrame("cdc_train"),
    model_id = "rf_2_2",
    ntrees = 150,
    max_depth = 20,
    mtries = -1,
    min_rows = 1,
    nfolds = 5,
    fold_assignment = "Modulo"
  )

# model summary
summary(rf_2_1)
summary(rf_2_2)

# glm의 변수중요도
h2o.varimp_plot(rf_2_1)
h2o.varimp_plot(rf_2_2)

# glm의 AUC
h2o.auc(rf_2_1)
h2o.auc(rf_2_2)

# Test set의 Target 값 가져오기
target_vec_test <- h2o.getFrame("cdc_test")$Class %>% as.vector()

# Test set의 예측값 계산
pred_rf_2_1 <- rf_2_1 %>% h2o.predict(h2o.getFrame("cdc_test")) %>% as.data.table()
pred_rf_2_2 <- rf_2_2 %>% h2o.predict(h2o.getFrame("cdc_test")) %>% as.data.table()
pred_rf_2_1[, "Class" := target_vec_test]
pred_rf_2_2[, "Class" := target_vec_test]

# Accuracy 측정
pred_rf_2_1[, mean(Class == predict)]
pred_rf_2_1[Class == 0, mean(Class == predict)]
pred_rf_2_1[Class == 1, mean(Class == predict)]

pred_rf_2_2[, mean(Class == predict)]
pred_rf_2_2[Class == 0, mean(Class == predict)]
pred_rf_2_2[Class == 1, mean(Class == predict)]

# PR ROC curve 그려보기
fg <- pred_rf_2_1[Class == 1, p1]
bg <- pred_rf_2_1[Class == 0, p1]

pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
print(pr)
plot(pr)

fg <- pred_rf_2_2[Class == 1, p1]
bg <- pred_rf_2_2[Class == 0, p1]

pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
print(pr)
plot(pr)

## 4) RandomForest - grid search
hyper_params <- list(
  ntrees = c(50, 100, 150, 200, 250, 300),
  max_depth = 20,
  mtries = as.integer(seq(sqrt(length(features)), (length(features)/3), length.out = 5))
)

rf_gridss <-
  h2o.grid(
    x = features,
    y = target,
    algorithm = "randomForest",
    training_frame = h2o.getFrame("cdc_train"),
    grid_id = "rf_gridss",
    hyper_params = hyper_params
    # nfolds = 5,
    # fold_assignment = "Modulo"
  )

rf_grids_auc <- h2o.getGrid("rf_gridss", "auc", TRUE)
best_rf <- h2o.getModel(rf_grids_auc@model_ids[[1]])

# Test set의 예측값 계산
pred_rf_best <- best_rf %>% h2o.predict(h2o.getFrame("cdc_test")) %>% as.data.table()
pred_rf_best[, "Class" := target_vec_test]

pred_rf_best[, mean(Class == predict)]
pred_rf_best[Class == 0, mean(Class == predict)]
pred_rf_best[Class == 1, mean(Class == predict)]

# PR ROC curve 그려보기
fg <- pred_rf_best[Class == 1, p1]
bg <- pred_rf_best[Class == 0, p1]

pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
print(pr)
plot(pr)


##########################################
# 2. Anomaly Detection using autoencoder #
##########################################

cdc_train <- h2o.getFrame("cdc_train")
cdc_train_0 <- cdc_train[which(as.vector(cdc_train$Class) == 0),]

autoencoder <- h2o.deeplearning(
  x = features,
  training_frame = cdc_train_0,
  model_id = "autoencoder",
  activation = "Tanh",
  export_weights_and_biases = TRUE,
  variable_importances = TRUE,
  autoencoder = TRUE, 
  hidden = c(100, 20, 100), epochs = 20
)

cdc_test <- h2o.getFrame("cdc_test")
ae_pred_test <- as.data.table(h2o.predict(autoencoder, cdc_test))
cdc_test_dt <- as.data.table(cdc_test)

ae_pred_test[, Class := cdc_test_dt[, Class]]

setnames(ae_pred_test, names(ae_pred_test), c(features, target))
dt_for_mse <- ae_pred_test[, features, with = F] - cdc_test_dt[, features, with = F]

reconstr_mse <- apply(dt_for_mse, 1, function(x) mean(x^2))
cdc_test_dt[, R_mse := reconstr_mse]

ggplot(cdc_test_dt, aes(x = 1:nrow(cdc_test_dt), y = R_mse)) + 
  geom_point(aes(colour = Class))

r_mse_1_mean <- cdc_test_dt[Class == 1, mean(R_mse)]
r_mse_1_quantiles <- cdc_test_dt[Class == 1, quantile(R_mse, c(0, .25, .50, .75, 1.))]
r_mse_1_quantiles

cdc_test_dt[, pred_by_mean := as.factor(ifelse(R_mse >= r_mse_1_mean, 1, 0))]
cdc_test_dt[, pred_by_25 := as.factor(ifelse(R_mse >= r_mse_1_quantiles[2], 1, 0))]

cdc_test_dt[, mean(Class == pred_by_mean)]
cdc_test_dt[, mean(Class == pred_by_25)]

cdc_test_dt[Class == 0, mean(Class == pred_by_mean)]
cdc_test_dt[Class == 1, mean(Class == pred_by_mean)]

cdc_test_dt[Class == 0, mean(Class == pred_by_25)]
cdc_test_dt[Class == 1, mean(Class == pred_by_25)]

# autoencoder의 가중치를 이용한 deep learning 모델 생성하기
dl_by_ae <- h2o.deeplearning(
  x = features,
  y = target,
  training_frame = h2o.getFrame("cdc_train"),
  model_id = "dl_by_ae",
  activation = "Tanh",
  export_weights_and_biases = TRUE,
  variable_importances = TRUE,
  hidden = c(100, 20, 100),
  epochs = 10, 
  pretrained_autoencoder = "autoencoder"
)

pred_dl_by_ae <- as.data.table(h2o.predict(dl_by_ae, cdc_test))
pred_dl_by_ae[, Class := as.vector(cdc_test$Class)]

pred_dl_by_ae[, mean(Class == predict)]
pred_dl_by_ae[Class == 0, mean(Class == predict)]
pred_dl_by_ae[Class == 1, mean(Class == predict)]

# PR ROC curve 그려보기
fg <- pred_dl_by_ae[Class == 1, p1]
bg <- pred_dl_by_ae[Class == 0, p1]

pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
print(pr)
plot(pr)

