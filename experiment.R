# load data --------------------------------------------------------------------
da_iris <- iris

# implementing models ----------------------------------------------------------
## Adaline
### adaline model with sigmoid activation
adaline <- function(formula, data, epochs = 500, lr = 0.05) {
  ## response in dummy format
  truth_form <- glue::glue('~ -1 + {all.vars(formula)[1]}')
  truth <- model.matrix(as.formula(truth_form), data = data)
  colnames(truth) <- stringr::str_remove(colnames(truth), all.vars(formula)[1])

  # matrix of weights
  ## ncol(truth) == n_class
  W <- matrix(rnorm(ncol(truth) * ncol(data)), ncol = ncol(truth))

  # double loop
  ## iterating first on rw (rows) and then on ep (epochs)
  for(ep in 1:epochs) {
    for(rw in 1:nrow(data)) {
      # data specification and prediction
      X <- model.matrix(formula, data = data[rw,]) # [1 Var1 Var2 ...] (1 x (p+1))
      Ui <- X %*% W # (1 x (p+1)) * ((p+1) x n_class) = (1 x n_class)
      Yi <- 1/(1 + exp(-Ui)) # sigmoid activation

      # error quantification
      Ei <- truth[rw,] - Yi

      # learning phase
      W <- W + lr*(t(X)/as.numeric(X%*%t(X)))%*%Ei
    }
  }
  # converting the function into a model like any other
  # already implemented in R
  model <- structure(list(W = W, formula = formula,
                          labels = colnames(truth)), class = "adaline")

  return(model)
}
### logistic perceptron predict function
predict.adaline <- function(object, newdata) {
  X <- model.matrix(object$formula, data = newdata) # [1 newdata]
  Ui <- X %*% object$W # (1 x (p+1)) * ((p+1) x n_class) = (1 x n_class)
  Yi <- 1/(1 + exp(-Ui)) # sigmoid activation
  estimate <- object$labels[max.col(Yi)] # get labels of the largest activation
  return(factor(estimate, levels = object$labels))
}

## PL
### logistic perceptron model
perceptron_log <- function(formula, data, epochs = 500, lr = 0.05, mom = 0.01) {
  ## response in dummy format
  truth_form <- glue::glue('~ -1 + {all.vars(formula)[1]}')
  truth <- model.matrix(as.formula(truth_form), data = data)
  colnames(truth) <- stringr::str_remove(colnames(truth), all.vars(formula)[1])

  # matrix of weights
  W <- matrix(rnorm(ncol(truth) * ncol(data)), ncol = ncol(truth))
  W_old <- W

  # double loop
  ## iterating first on rw (rows) and then on ep (epochs)
  for(ep in 1:epochs) {
    for(rw in 1:nrow(data)) {
      # data specification and prediction
      X <- model.matrix(formula, data = data[rw,]) # [1 Var1 Var2 ...] (1 x (p+1))
      Ui <- X %*% W # (1 x (p+1)) * ((p+1) x n_class) = (1 x n_class)
      Yi <- 1/(1 + exp(-Ui)) # sigmoid activation

      # error quantification
      Ei <- truth[rw,] - Yi
      Di <- 0.5 * (1 - Yi^2) + 0.05
      DDi <- Ei * Di

      # learning phase
      W_aux <- W
      W <- W + lr*t(X)%*%DDi + mom*(W - W_old)
      W_old <- W_aux
    }
  }
  # converting the function into a model like any other
  # already implemented in R
  model <- structure(list(W = W, formula = formula,
                          labels = colnames(truth)), class = "pl")

  return(model)
}
### logistic perceptron predict function
predict.pl <- function(object, newdata) {
  X <- model.matrix(object$formula, data = newdata) # [1 newdata]
  Ui <- X %*% object$W # (1 x (p+1)) * ((p+1) x n_class) = (1 x n_class)
  Yi <- 1/(1 + exp(-Ui)) # sigmoid activation
  estimate <- object$labels[max.col(Yi)] # get labels of the largest activation
  return(factor(estimate, levels = object$labels))
}

## LMQ
### LMQ model with tikhonov (lambda)
lmq <- function(formula, data, lambda = 1e-3) {
  # data specification
  X <- model.matrix(formula, data = data)[,-1] # no intercept (bias)
  # response (y) in dummy format
  y_form <- glue::glue('~ -1 + {all.vars(formula)[1]}')
  y <- model.matrix(as.formula(y_form), data = data)
  colnames(y) <- stringr::str_remove(colnames(y), all.vars(formula)[1])

  # get weigth matrix W
  W <- solve(t(X) %*% X + diag(lambda, ncol(X))) %*% t(X) %*% y

  # converting the function into a model like any other
  # already implemented in R
  model <- structure(list(W = W, formula = formula,
                          labels = colnames(y)), class = "lmq")

  return(model)
}

### LMQ predict function
predict.lmq <- function(object, newdata) {
  X <- model.matrix(object$formula, data = newdata)[,-1] # no intercept (bias)
  y_pred <- X %*% object$W # vector of scores for each discriminant
  estimate <- object$labels[max.col(y_pred)] # get labels of the largest score
  return(factor(estimate, levels = object$labels))
}

## MLP
### multilayer perceptron model
mlp <- function(formula, data, size = 10, epochs = 500, lr = 0.05, mom = 0.01) {
  ## response in dummy format
  truth_form <- glue::glue('~ -1 + {all.vars(formula)[1]}')
  truth <- model.matrix(as.formula(truth_form), data = data)
  colnames(truth) <- stringr::str_remove(colnames(truth), all.vars(formula)[1])

  # matrix of weights
  ## input layer
  W <- matrix(rnorm(size * ncol(data)), ncol = size)
  W_old <- W

  ## hidden layer
  H <- matrix(rnorm(ncol(truth) * (size + 1)), ncol = ncol(truth))
  H_old <- H

  # double loop
  ## iterating first on rw (rows) and then on ep (epochs)
  for(ep in 1:epochs) {
    for(rw in 1:nrow(data)) {
      # data specification and prediction
      X <- model.matrix(formula, data = data[rw,]) # [1 Var1 Var2 ...] (1 x (p+1))
      ## hidden layer
      Ui <- X %*% W # (1 x (p+1)) * ((p+1) x size) = (1 x size)
      Zi <- 1/(1 + exp(-Ui)) # sigmoid activation
      ## output layer
      Z <- cbind(1, Zi) # [1 Z1 Z2 ...] (1 x (size+1))
      Uk <- Z %*% H # (1 x (size+1)) * ((size+1) x n_class)
      Yk <- 1./(1+exp(-Uk)) # sigmoid activation

      # error quantification
      Ek <- truth[rw,] - Yk

      # local gradients
      ## output layer
      Dk <- Yk * (1 - Yk) + 0.01
      DDk <- Ek * Dk
      ## hidden layer
      Di <- Zi * (1 - Zi) + 0.01
      DDi <- Di * DDk %*% t(H[-1,])

      # learning phase
      ## output layer
      H_aux <- H
      H <- H + lr*t(Z)%*%DDk + mom*(H - H_old)
      H_old <- H_aux
      ## hidden layer
      W_aux <- W
      W <- W + 2*lr*t(X)%*%DDi + mom*(W - W_old)
      W_old <- W_aux
    }
  }
  # converting the function into a model like any other
  # already implemented in R
  model <- structure(list(W = W, H = H, formula = formula,
                          labels = colnames(truth)), class = "mlp")

  return(model)
}

### mlp predict function
predict.mlp <- function(object, newdata) {
  X <- model.matrix(object$formula, data = newdata) # [1 newdata]
  ## hidden layer
  Ui <- X %*% object$W # (1 x (p+1)) * ((p+1) x size) = (1 x size)
  Zi <- 1/(1 + exp(-Ui)) # sigmoid activation
  ## output layer
  Z <- cbind(1, Zi) # [1 Z1 Z2 ...] (1 x (size+1))
  Uk <- Z %*% object$H # (1 x (size+1)) * ((size+1) x n_class)
  Yk <- 1./(1+exp(-Uk)) # sigmoid activation
  estimate <- object$labels[max.col(Yk)] # get labels of the largest activation
  return(factor(estimate, levels = object$labels))
}

# useful functions -------------------------------------------------------------
## get metrics function
get_metrics <- function(models, da_test, truth) {
  purrr::map_dfr(models, function(model) {
    estimate <- predict(model, da_test) # prediction
    cm <- table(truth, estimate) # confusion matrix
    # get metrics
    accuracy <- sum(diag(cm))/sum(cm)
    precision_by_class <- diag(cm)/colSums(cm)
    precision_by_class[is.nan(precision_by_class)] <- 0 # fix division by zero
    # store metrics in a dataframe
    metrics <- tibble::tibble(accuracy) |>
      tibble::add_column(dplyr::bind_rows(precision_by_class))
    return(metrics)
  }, .id = 'model')
}

# run experiment ---------------------------------------------------------------
## settings for parallel processing (multiple iterations at the same time)
globals <- list('da_iris' = da_iris, 'get_metrics' = get_metrics,
                'adaline' = adaline, 'perceptron_log' = perceptron_log,
                'lmq' = lmq, 'mlp' = mlp, 'predict.adaline' = predict.adaline,
                'predict.pl' = predict.pl, 'predict.lmq' = predict.lmq,
                'predict.mlp' = predict.mlp)
future::plan(future::multisession, workers = 25) # 25 iterations at the same time
## loop 100x and assign results to `da_experiment`
da_experiment <- furrr::future_map_dfr(1:100, function(seed) {
  ## data split 80/20 (train/test)
  set.seed(seed)
  da_split <- rsample::initial_split(da_iris, prop = 0.8, strata = "Species")
  da_train <- rsample::training(da_split)
  da_test <- rsample::testing(da_split)

  ## apply models in train
  mod_ada <- adaline(formula = Species ~ ., data = da_train)
  mod_pl <- perceptron_log(formula = Species ~ ., data = da_train)
  mod_lmq <- lmq(formula = Species ~ ., data = da_train)
  mod_mlp <- mlp(formula = Species ~ ., data = da_train)

  ## collect metrics in test
  metrics <- get_metrics(models = list('ada' = mod_ada, 'pl' = mod_pl,
                                       'lmq' = mod_lmq, 'mlp' = mod_mlp),
                         da_test = da_test, truth = da_test[, "Species"])
  return(metrics)
}, .id = 'seed', .progress = TRUE,
.options = furrr::furrr_options(seed = TRUE, globals = globals))

# write da_metrics in a .csv file
fs::dir_create('data')
readr::write_csv(da_experiment, 'data/experiment.csv')
