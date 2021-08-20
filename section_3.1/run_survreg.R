library(survival)
library(rjson)

source('data_importer.R')

data_dir <- '../data'

get.accuracy <- function(pred, y) {
  res <- (pred >= y[, "min.log.penalty"] & pred <= y[,"max.log.penalty"])
  accurcy <- sum(res) / length(res)
  return(accurcy)
}

get.pca.transform <- function(pca, data) {
  pred.pr <- predict(pca, newdata=data)
  return(pred.pr)
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

sr.predict <- function(clf, X) {
  pred <- predict(clf, X)
  return(pred)
}

sr.fit <- function(X,y) {
  data <- data.frame(X, y)
  my.surv <- Surv(y[, "min.log.penalty"], y[, "max.log.penalty"], type='interval2')
  formula <- as.formula(paste("my.surv ~", paste(colnames(X), collapse="+")))
  fit <- try(survreg(formula, data=data, dist='gaussian', control=list(maxiter=1000)))
  return(fit)
}

sr.pca <- function(train, test, y.trn, y.tst){
  no.var.cols <- getNonVarCols(train)
  train <- train[ , !(colnames(train) %in% no.var.cols)]
  test <- test[ , !(colnames(test) %in% no.var.cols)]
  train.pr <- prcomp(train, center=TRUE, scale=TRUE)
  nPr <- length(train.pr$sdev)
  accuracy <- numeric(nPr)
  for(j in 1:nPr) {
    train.sub <- as.matrix(train.pr$x[, c(1:j)])
    colnames(train.sub) <- paste("PC", c(1:j), sep="")
    sr.pca.fit <- sr.fit(train.sub, y.trn)
    if(class(sr.pca.fit) != "try-error"){
      pred.pr <- get.pca.transform(train.pr, test)
      pred.pr.sub <- data.frame(pred.pr[, c(1:j)])
      colnames(pred.pr.sub) <- paste("PC", c(1:j), sep="")
      pred.y <- predict(sr.pca.fit ,pred.pr.sub)
      accuracy[j] <- get.accuracy(pred.y, y.tst)
    }
  }
  return(accuracy)
}

sr.cv <- function(X,y){
  train.folds <- cut(seq(1,nrow(X)), breaks=5, labels=FALSE)
  res <- list()
  for(j in 1:5) {
    testIndexes <- which(train.folds==j, arr.ind=TRUE)
    X.tst <- X[testIndexes,]
    X.trn <- X[-testIndexes,]
    y.trn <- y[-testIndexes,]
    y.tst <- y[testIndexes,]
    res[[j]] <- sr.pca(X.trn, X.tst, y.trn, y.tst)
  }
  result <- do.call(cbind,res)
  result <- cbind(result,apply(result,1,mean))
  pr_sel <- which.max(result[,6])
  X.pr <- prcomp(X, center=TRUE, scale=TRUE)
  X.sub <- X.pr$x[,1:pr_sel]
  if(class(X.sub) == 'matrix' | class(X.sub) == 'numeric') {
    X.sub <- data.frame(X.sub)
  }
  colnames(X.sub) <- paste("PC", c(1:pr_sel), sep="")
  print(pr_sel)
  sr.cv.fit <- sr.fit(X.sub,y)
  result <- list()
  result$fit <- sr.cv.fit
  result$pr_sel <- pr_sel
  result$pr <- X.pr
  return(result)
}

main <- function(args) {
  if (length(args) != 2) {
    stop("Usage: Rscript run_survreg.R [data_source] [dataset]")
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

    result <- sr.cv(X.train, y.train)
    pr <- result$pr
    sr.cv.fit <- result$fit
    pr_sel <- result$pr_sel
    X.test.pr <- get.pca.transform(pr, X.test)
    X.test.pr <- X.test.pr[, 1:pr_sel]
    if(class(X.test.pr) == 'matrix' | class(X.test.pr) == 'numeric'){
      X.test.pr <- data.frame(X.test.pr)
    }
    colnames(X.test.pr) <- paste("PC", c(1:pr_sel), sep="")
    y.pred.test <- sr.predict(sr.cv.fit, X.test.pr)
    accuracy_fold[[i]] <- get.accuracy(y.pred.test, y.test)
    end_time <- Sys.time()
    time_taken <- as.numeric(end_time - start_time)
    run_time[[i]] <- time_taken
  }
  names(accuracy_fold) <- fold_iter
  names(run_time) <- fold_iter
  json.accuracy <- toJSON(accuracy_fold)
  json.runtime <- toJSON(run_time)
  filename <- file.path('survreg-log', dataset, 'accuracy.json')
  write(json.accuracy, file=filename)
  filename <- file.path('survreg-log', dataset, 'run_time.json')
  write(json.runtime, file=filename)
}

args <- commandArgs(trailingOnly=TRUE)
main(args)
