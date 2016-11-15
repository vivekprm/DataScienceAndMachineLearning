# Multiple Linear Regression

# Data Preprocessing

# Importing dataset
dataset = read.csv('50_Startups.csv')
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set =  subset(dataset, split == TRUE)
test_set =  subset(dataset, split == FALSE)
# Feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

# Fitting Multiple Linear Regression to Training set
# regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State)
# Same as above
regressor = lm(formula = Profit ~ .,
               data = training_set)

# By looking at the summary(regressor) only siginificant variable is R.D.Spend so we can reduce
# the above equation into simple linear regression. i.e. regressor = lm(formula = Profit ~ R.D.Spend, = training_set)

# Predicting the test set results.
Y_pred = predict(regressor, newdata = test_set)

# Check Y_pred and compare Y_pred with only significant variable. It'll almost be the same.
# regressor1 = lm(formula = Profit ~ R.D.Spend,
#               data = training_set)
# Y_pred = predict(regressor1, newdata = test_set)

# Building optimal model using Backward Elimination.
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = training_set)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = training_set)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = training_set)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend,
               data = training_set)
summary(regressor)


