# load data --------------------------------------------------------------------
da_iris <- iris

# implementing models ----------------------------------------------------------
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
      Ui <- X %*% W # (1 x (p+1)) * ((p+1) x size) = (1 x size)
      Yi <- 1/(1 + exp(-Ui)) # sigmoid activation

      # error quantification
      Ei <- truth[rw,] - Yi
      Di <- 0.5 * (1 - Yi^2) + 0.05
      DDi <- Ei * Di

      # learning phase
      W_aux <- W
      W <- W + lr*t(X)%*%DDi
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
  Ui <- X %*% object$W
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

# useful functions -------------------------------------------------------------
## get metrics function
get_metrics <- function(models, da_test, truth) {
  purrr::map_dfr(models, function(model) {
    estimate <- predict(model, da_test)
    cm <- table(truth, estimate)
    accuracy <- sum(diag(cm))/sum(cm)
    precision_by_class <- diag(cm)/colSums(cm)
    precision_by_class[is.nan(precision_by_class)] <- 0 # fix division by zero
    metrics <- tibble::tibble(accuracy) |>
      tibble::add_column(dplyr::bind_rows(precision_by_class))
    return(metrics)
  }, .id = 'model')
}

# run experiment ---------------------------------------------------------------
## loop 100x and assign results to `da_experiment`
da_experiment <- purrr::map_dfr(1:3, function(seed) {
  ## data split 80/20 (train/test)
  set.seed(seed)
  da_split <- rsample::initial_split(da_iris, prop = 0.8, strata = "Species")
  da_train <- rsample::training(da_split)
  da_test <- rsample::testing(da_split)

  ## apply models in train
  mod_ada <- nnet::multinom(formula = Species ~ ., data = da_train, trace = FALSE)
  mod_pl <- perceptron_log(formula = Species ~ ., data = da_train)
  mod_lmq <- lmq(formula = Species ~ ., data = da_train)
  mod_mlp <- nnet::multinom(formula = Species ~ ., data = da_train, trace = FALSE)

  ## collect metrics in test
  metrics <- get_metrics(models = list('ada' = mod_ada, 'pl' = mod_pl,
                                       'lmq' = mod_lmq, 'mlp' = mod_mlp),
                         da_test = da_test, truth = da_test[, "Species"])

  return(metrics)
}, .id = 'seed')
