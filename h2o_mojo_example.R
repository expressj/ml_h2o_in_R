library(h2o)
h2o.init(nthreads = -1)

path <- system.file("extdata", "prostate.csv", package="h2o")
h2o_df <- h2o.importFile(path, destination_frame = "prostate_h2o")
h2o_df$CAPSULE <- as.factor(h2o_df$CAPSULE)

x <- c("AGE", "RACE", "PSA", "GLEASON")
y <- "CAPSULE"

prostate_gbm <- h2o.gbm(x=x, y=y, training_frame=h2o_df,
                 distribution="bernoulli", model_id = "prostate_gbm",
                 ntrees=100, max_depth=4, learn_rate=0.1)

if (!dir.exists("./experiments"))
  dir.create("./experiments")

modelfile <- h2o.download_mojo(prostate_gbm, path = "./experiments", get_genmodel_jar = TRUE)
print(paste0("Model saved to ", normalizePath("./experiments/"), modelfile))