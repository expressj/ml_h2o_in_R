library(h2o)
h2o.init(nthreads = -1)

train_file <- "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/train.csv.gz"
test_file <- "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/test.csv.gz"
train <- h2o.importFile(train_file)
test <- h2o.importFile(test_file)

head(train)

y <- "C785"
x <- setdiff(names(train), y)
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])

# 약5분 : 6코어 12쓰레드 3.30GHz 기준
model <- h2o.deeplearning(x = x, y = y, training_frame = train, validation_frame = test, distribution = "multinomial", 
                          activation = "Rectifier", hidden = c(200, 200), epochs = 1000, model_id = "mnist_dnn")
summary(model)
# Model 저장
h2o.saveModel(model, "./mnist_dnn")

# Model 불러오기
load_model <- h2o.loadModel("./mnist_dnn/mnist_dnn")





