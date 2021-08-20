library(survival)
library(penaltyLearning)
library(rjson)

source('data_importer.R')

data_dir <- '../data'

get.accuracy <- function(pred, y) {
  res <- (pred >= y[,"min.log.penalty"] & pred <= y[,"max.log.penalty"])
  acc <- sum(res) / length(res) 
  return(acc)
}


get.train.test <- function(folds, fold_id, X, y) {
  X.train <- X[folds != fold_id,]
  X.test <- X[folds == fold_id,]
  y.train <- y[folds != fold_id,]
  y.test <- y[folds == fold_id,]
  res <- list()
  res$X.train <- X.train
  res$X.test <- X.test
  res$y.train <- y.train
  res$y.test <- y.test
  return(res)
}

main <- function(args) {
  if (length(args) != 2) {
    stop("Usage: Rscript run_penalty_learning.R [data_source] [dataset]")
  }
  data_source <- args[1]
  dataset <- args[2]

  print(dataset)
  result <- get.data(data_dir, data_source, dataset)
  X <- result$X
  y <- result$y
  folds <- result$folds$fold
  fold_iter <- unique(folds)
  accuracy_fold <- list()
  run_time <- list()
  for (i in fold_iter) {
    start_time <- Sys.time()
    res <- get.train.test(folds, i, X, y)
    X.train <- res$X.train
    X.test <- res$X.test
    y.train <- res$y.train
    y.test <- res$y.test
    if (class(X.train) == 'data.frame') {
      X.train <- as.matrix(X.train)
    }
    if (class(y.train) == 'data.frame') {
      y.train <- as.matrix(y.train)
    }
    if (class(X.test) == 'data.frame') {
      X.test <- as.matrix(X.test)
    }
    if (class(y.test) == 'data.frame') {
      y.test <- as.matrix(y.test)
    }
    model <- IntervalRegressionCV(X.train, y.train)  
    y.pred.test <- predict(model, X.test)
    accuracy_fold[[i]] <- get.accuracy(y.pred.test, y.test)
    end_time <- Sys.time()
    time_taken <- as.numeric(end_time-start_time)
    run_time[[i]] <- time_taken
  }
  names(accuracy_fold) <- fold_iter
  names(run_time) <- fold_iter
  json.accuracy <- toJSON(accuracy_fold)
  json.runtime <- toJSON(run_time)
  filename <- file.path('penaltyLearning-log', dataset, 'accuracy.json')
  write(json.accuracy, file=filename)
  filename <- file.path('penaltyLearning-log', dataset, 'run_time.json')
  write(json.runtime, file=filename)
}

args <- commandArgs(trailingOnly=TRUE)
main(args)
