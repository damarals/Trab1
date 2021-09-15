# load data --------------------------------------------------------------------
coln <- 24
u <- glue::glue('https://archive.ics.uci.edu/ml/machine-learning-databases/00194/sensor_readings_{coln}.data')
da_robot_colnames <- c(paste0('us', 1:coln), 'class')
da_robot <- readr::read_csv(file = u, col_names = da_robot_colnames,
                            col_types = paste0(c(rep('d', coln), 'c'), collapse = ""))

# preprocessing ----------------------------------------------------------------
da_robot <- da_robot |>
  dplyr::mutate_if(rlang::is_double,
                   function(x) (x - min(x))/(max(x) - min(x))) |>
  dplyr::mutate(class = stringr::str_replace_all(class, '-', '_'),
                class = stringr::str_to_lower(class),
                class = factor(class, levels = c('move_forward', 'slight_right_turn',
                                                 'sharp_right_turn', 'slight_left_turn')))

# implementing models ----------------------------------------------------------
## Adaline
### adaline model with sigmoid activation
adaline <- function(formula, data, epochs = 200, lr = 0.1) {
  # matrix of weights
  nclass <- nrow(dplyr::distinct(data[, all.vars(formula)[1]]))
  W <- matrix(rnorm(nclass * ncol(data)), ncol = nclass)

  # double loop
  ## iterating first on rw (rows) and then on ep (epochs)
  for(ep in 1:epochs) {
    ## shuffle data
    data <- data[sample(1:nrow(data), nrow(data), replace = FALSE), ]
    ## response in dummy format
    truth_form <- glue::glue('~ -1 + {all.vars(formula)[1]}')
    truth <- model.matrix(as.formula(truth_form), data = data)
    colnames(truth) <- stringr::str_remove(colnames(truth), all.vars(formula)[1])
    ## metrics vars
    best_EQ <- 0 # for best epoch based on EQ
    EQ <- 0 # store EQ for each epoch
    for(rw in 1:nrow(data)) {
      # data specification and prediction
      X <- model.matrix(formula, data = data[rw,]) # [1 Var1 Var2 ...] (1 x (p+1))
      Ui <- X %*% W # (1 x (p+1)) * ((p+1) x n_class) = (1 x n_class)
      Yi <- 1/(1 + exp(-Ui)) # sigmoid activation
      # error quantification
      Ei <- truth[rw,] - Yi
      EQ <- EQ + 0.5*sum(Ei^2)
      # learning phase
      W <- W + lr*(t(X)/as.numeric(X%*%t(X)))%*%Ei
    }
    # best epoch verification
    if(EQ > best_EQ) {
      best_epoch <- list(W = W)
      best_eq <- EQ
    }
    #message(glue::glue('Epoch {ep}, MSE: {round(EQ/nrow(data), 5)}'))
  }
  # converting the function into a model like any other
  # already implemented in R
  model <- structure(list(W = best_epoch$W, formula = formula,
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
perceptron_log <- function(formula, data, epochs = 200, lr = 0.5, mom = 0.3) {
  # matrix of weights
  nclass <- nrow(dplyr::distinct(data[, all.vars(formula)[1]]))
  W <- matrix(rnorm(nclass * ncol(data)), ncol = nclass)
  W_old <- W

  # double loop
  ## iterating first on rw (rows) and then on ep (epochs)
  for(ep in 1:epochs) {
    ## shuffle data
    data <- data[sample(1:nrow(data), nrow(data), replace = FALSE), ]
    ## response in dummy format
    truth_form <- glue::glue('~ -1 + {all.vars(formula)[1]}')
    truth <- model.matrix(as.formula(truth_form), data = data)
    colnames(truth) <- stringr::str_remove(colnames(truth), all.vars(formula)[1])
    ## metrics vars
    best_EQ <- 0 # for best epoch based on EQ
    EQ <- 0 # store EQ for each epoch
    for(rw in 1:nrow(data)) {
      # data specification and prediction
      X <- model.matrix(formula, data = data[rw,]) # [1 Var1 Var2 ...] (1 x (p+1))
      Ui <- X %*% W # (1 x (p+1)) * ((p+1) x n_class) = (1 x n_class)
      Yi <- 1/(1 + exp(-Ui)) # sigmoid activation

      # error quantification
      Ei <- truth[rw,] - Yi
      EQ <- EQ + 0.5*sum(Ei^2)

      # local gradients
      Di <- 0.5 * (1 - Yi^2) + 0.05
      DDi <- Ei * Di

      # learning phase
      W_aux <- W
      W <- W + lr*t(X)%*%DDi + mom*(W - W_old)
      W_old <- W_aux
    }
    # best epoch verification
    if(EQ > best_EQ) {
      best_epoch <- list(W = W)
      best_eq <- EQ
    }
    #message(glue::glue('Epoch {ep}, MSE: {round(EQ/nrow(data), 5)}'))
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
mlp <- function(formula, data, size = 64, epochs = 300, lr = 0.2, mom = 0.4) {
  # matrix of weights
  ## input layer
  W <- matrix(rnorm(size * ncol(data)), ncol = size)
  W_old <- W

  ## hidden layer
  nclass <- nrow(dplyr::distinct(data[, all.vars(formula)[1]]))
  H <- matrix(rnorm(nclass * (size + 1)), ncol = nclass)
  H_old <- H

  # double loop
  ## iterating first on rw (rows) and then on ep (epochs)
  for(ep in 1:epochs) {
    ## shuffle data
    data <- data[sample(1:nrow(data), nrow(data), replace = FALSE), ]
    ## response in dummy format
    truth_form <- glue::glue('~ -1 + {all.vars(formula)[1]}')
    truth <- model.matrix(as.formula(truth_form), data = data)
    colnames(truth) <- stringr::str_remove(colnames(truth), all.vars(formula)[1])
    ## metrics vars
    best_EQ <- 0 # for best epoch based on EQ
    EQ <- 0 # store EQ for each epoch
    for(rw in 1:nrow(data)) {
      # data specification and prediction
      X <- model.matrix(formula, data = data[rw,]) # [1 Var1 Var2 ...] (1 x (p+1))
      ## hidden layer
      Ui <- X %*% W # (1 x (p+1)) * ((p+1) x size) = (1 x size)
      Zi <- 1/(1 + exp(-Ui)) # sigmoid activation
      # Zi <- exp(Ui)/sum(exp(Ui))
      ## output layer
      Z <- cbind(1, Zi) # [1 Z1 Z2 ...] (1 x (size+1))
      Uk <- Z %*% H # (1 x (size+1)) * ((size+1) x n_class)
      Yk <- 1/(1+exp(-Uk)) # sigmoid activation
      # Yk <- exp(Uk)/sum(exp(Uk))
      # error quantification
      Ek <- truth[rw,] - Yk
      EQ <- EQ + 0.5*sum(Ek^2)
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
    # best epoch verification
    if(EQ > best_EQ) {
      best_epoch <- list(W = W, H = H)
      best_eq <- EQ
    }
    #message(glue::glue('Epoch {ep}, MSE: {round(EQ/nrow(data), 5)}'))
  }
  # converting the function into a model like any other
  # already implemented in R
  model <- structure(list(W = best_epoch$W, H = best_epoch$H, formula = formula,
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
  Yk <- 1/(1 + exp(-Uk)) # sigmoid activation
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
globals <- list('da_robot' = da_robot, 'get_metrics' = get_metrics,
                'adaline' = adaline, 'perceptron_log' = perceptron_log,
                'lmq' = lmq, 'mlp' = mlp, 'predict.adaline' = predict.adaline,
                'predict.pl' = predict.pl, 'predict.lmq' = predict.lmq,
                'predict.mlp' = predict.mlp)
future::plan(future::multisession, workers = 5) # 10 iterations at the same time
## loop 100x and assign results to `da_experiment`
da_experiment <- furrr::future_map_dfr(1:100, function(seed) {
  ## data split 80/20 (train/test)
  set.seed(seed)
  da_split <- rsample::initial_split(da_robot, prop = 0.8, strata = "class")
  da_train <- rsample::training(da_split)
  da_test <- rsample::testing(da_split)

  ## apply models in train
  mod_ada <- adaline(formula = class ~ ., data = da_train)
  mod_pl <- perceptron_log(formula = class ~ ., data = da_train)
  mod_lmq <- lmq(formula = class ~ ., data = da_train)
  mod_mlp <- mlp(formula = class ~ ., data = da_train)

  ## collect metrics in test
  metrics <- get_metrics(models = list('ada' = mod_ada, 'pl' = mod_pl,
                                       'lmq' = mod_lmq, 'mlp' = mod_mlp),
                         da_test = da_test, truth = da_test$class)
  return(metrics)
}, .id = 'seed', .progress = TRUE,
.options = furrr::furrr_options(seed = TRUE, globals = globals))

# write da_metrics in a .csv file
fs::dir_create('data')
readr::write_csv(da_experiment, 'data/experiment.csv')
