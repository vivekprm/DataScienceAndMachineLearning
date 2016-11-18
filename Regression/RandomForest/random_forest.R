# RandomForest Regression

# Importing dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into Training set and test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set =  subset(dataset, split == TRUE)
# test_set =  subset(dataset, split == FALSE)

# Feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

# Fitting Regression model to Dataset
# install.packages('randomForest')
# library('randomForest')
# dataset[1] returns array while dataset$Salary returns vector

set.seed(1234)
regressor = randomForest(x = dataset[1],
                         y = dataset$Salary,
                         ntree = 500)

# Predicting a new result using RandomForest Regression
Y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualizing the RandomForest Regression results (For higher resolution and Smooth curve).
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (RandomForest Regression)') +
  xlab('Level') +
  ylab('Salary')