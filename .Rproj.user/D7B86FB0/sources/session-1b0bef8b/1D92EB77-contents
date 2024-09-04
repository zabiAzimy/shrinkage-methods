#==============================================================================#
# Shrinkage Regression: mtcars data set                                        #
#==============================================================================#
rm(list = ls(all.names = TRUE))

# 00: packages -----------------------------------------------------------------
library(glmnet)
library (ISLR2)

# 01: data preparation ---------------------------------------------------------
data("mtcars")
summary(mtcars)

# 03: data exploration ---------------------------------------------------------
hist(mtcars$mpg, breaks = 30, col = "lightblue")
plot(mtcars$hp, mtcars$mpg, col = "brown", lwd = 2)

# 02: ridge regression ---------------------------------------------------------
x <- model.matrix(mpg ~ hp + I(hp^2) + cyl + disp + drat + wt + qsec + vs +
                    am + gear + carb,
                  data = mtcars)[, -1]
y <- mtcars$mpg
grid <- 10^seq(10, -2, length = 100)
grid
ridge_mod <- glmnet(x, y, alpha = 0, lambda = grid)
dim(coef(ridge_mod ))

# check what kind of values are stored in ridge_mod$lambda
ridge_mod$lambda[50]

# strong penalization
ridge_mod$lambda[50]
coef(ridge_mod)[, 50]
sqrt(sum(coef(ridge_mod)[-1, 50]^2))

# weaker penalization
ridge_mod$lambda[60]
coef(ridge_mod)[, 60]
sqrt(sum(coef(ridge_mod)[-1, 60]^2))

# Add plots for change in coeff wrt lambda
plot(grid, coef(ridge_mod)["hp", ], log = "x", type = "l", xlab = "lambda", col = "blue", lwd=2)
plot(grid, coef(ridge_mod)["qsec", ], log = "x", type = "l", xlab = "lambda", col = "orange", lwd=2)
plot(grid, coef(ridge_mod)["cyl", ], log = "x", type = "l", xlab = "lambda", col = "red", lwd=2)
plot(grid, coef(ridge_mod)["(Intercept)", ], log = "x", type = "l", xlab = "lambda", col = "darkgreen", lwd=2)

# All together except the intercept
matplot(grid, t(coef(ridge_mod)[-1, ]), type = "l", log = "x", xlab = "lambda", lwd = 2)
# If you plot ridge_mod then you get a very similar plot 
plot(ridge_mod,xvar="lambda", lwd = 2)

# Predictions
predict(ridge_mod, s = 50, type ="coefficients")[1:11, ]

# Train/Test split
set.seed (1)
train <- sample (1: nrow(x), nrow(x)/2)
test <- (-train)
y_test <- y[test]

ridge_mod <- glmnet(x[train, ], y[train], alpha = 0, lambda = grid, thresh = 1e-12)
ridge_pred <- predict(ridge_mod, s = 4, newx = x[test, ])

# Test: MSE
mean((ridge_pred - y_test)^2)

# Intercept Model: MSE
mseNullModel <- mean(( mean(y[train ])-y_test)^2)
mseNullModel

# Test-MSE for very large penalization
ridge_pred <- predict (ridge_mod , s= 1e10, newx = x[test, ])
mean((ridge_pred - y_test)^2)

# Comparison with OLS (Oridinary Least Squares) regression:
ridge_pred <- predict(ridge_mod, s = 0, newx = x[test, ], exact = TRUE, 
                      x = x[train, ], y = y[train])
mseFullModel <- mean((ridge_pred - y_test)^2)
mseFullModel
lm(y ~ x, subset =train)$coefficients
predict(ridge_mod, s = 0, exact = TRUE, x = x[train, ], 
        y = y[train], type = "coefficients")[1:11, ]

# 02b: Ridge CV: ---------------------------------------------------------------
set.seed (1)
cv_out <- cv.glmnet(x[train, ], y[train], alpha = 0)
plot(cv_out)
bestlam <- cv_out$lambda.min
bestlam

ridge_pred <- predict(ridge_mod, s = bestlam, newx = x[test, ])
mseBestLam <- mean((ridge_pred - y_test)^2)
mseBestLam

out <- glmnet (x, y, alpha = 0)
predict(out, type = "coefficients", s = bestlam )[1:11, ]

# 03: Lasso --------------------------------------------------------------------
lasso_mod <- glmnet(x[train, ], y[train], alpha = 1,lambda = grid)
plot(lasso_mod, lwd =2)

# 03b: Lasso CV: ---------------------------------------------------------------
set.seed (1)
cv_out <- cv.glmnet(x[train, ], y[train], alpha = 1)
plot(cv_out)
bestlam <- cv_out$lambda.min
bestlam
lasso.pred <- predict(lasso_mod, s = bestlam, newx = x[test, ])
mean((lasso.pred - y_test)^2)

out <- glmnet(x, y, alpha = 1, lambda = grid)
lasso_coef <- predict(out, type = "coefficients", s = bestlam)[2:11, ]
lasso_coef
varLasso <- lasso_coef[lasso_coef !=0]


