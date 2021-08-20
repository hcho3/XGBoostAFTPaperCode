
getNonVarCols <- function(data) {
  var_columns <- apply(data, 2, var)
  resCol <- names(var_columns[var_columns == 0.0])
  return(resCol)
}

get.data <- function(data_dir, data_source, dataset) {
  dataset_dir <- file.path(data_dir, dataset)
  if (data_source == "chipseq") {
    inputFileName <- file.path(dataset_dir, 'inputs.csv')
    labelFileName <- file.path(dataset_dir, 'outputs.csv')
    foldsFileName <- file.path(dataset_dir, 'cv/equal_labels/folds.csv')
    inputs <- read.table(inputFileName, sep=",", header=TRUE, stringsAsFactors=FALSE, row.names=1)
    labels <- read.table(labelFileName, sep=",", header=TRUE, stringsAsFactors=FALSE, row.names=1)
    colnames(labels) <- c('min.log.penalty', 'max.log.penalty')
    folds <- read.table(foldsFileName, sep=",", header=TRUE, stringsAsFactors=FALSE, row.names=1)

    rownamesInput <- rownames(inputs)
    inputs <- do.call(data.frame,lapply(inputs, function(x) replace(x, is.infinite(x), NA)))
    naColumns <- colnames(inputs)[colSums(is.na(inputs)) > 0]
    noVarCol <- getNonVarCols(inputs)
    removeCols <- c(naColumns,noVarCol)
    inputs <- inputs[, !(colnames(inputs) %in% removeCols)]
    rownames(inputs) <- rownamesInput

    X <- inputs
    y <- labels
  } else if (data_source == "simulated") {
    X_path <- file.path(dataset_dir, 'features.csv')
    y_path <- file.path(dataset_dir, 'targets.csv')
    folds_path <- file.path(dataset_dir, 'folds.csv')
    X <- read.table(X_path, sep=",", header=TRUE, stringsAsFactors=FALSE)
    y <- read.table(y_path, sep=",", header=TRUE, stringsAsFactors=FALSE)
    folds <- read.table(folds_path, sep=",", header=TRUE, stringsAsFactors=FALSE)
  } else {
    stop("data_source must be one of {chipseq,simulated}")
  }
  result <- list()
  result$X <- X
  result$y <- y
  result$folds <- folds
  return(result)
}
