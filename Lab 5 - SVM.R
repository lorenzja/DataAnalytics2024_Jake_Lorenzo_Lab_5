library(ggfortify)
library(e1071)
library(class)
library(psych)
library(ggplot2)
library(caret)

names(wine) <- c("Type","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid Phenols","Proanthocyanins","Color Intensity","Hue","Od280/od315 of diluted wines","Proline")
head(wine)

wine$Type <- as.factor(wine$Type)

pairs.panels(wine[,-1],gap = 0,bg = c("red", "yellow", "blue")[wine$Type],pch=21)

# ## split train/test
train.indexes <- sample(177,0.7*177)

train <- wine[train.indexes,]
test <- wine[-train.indexes,]

## separate x (features) & y (class labels)
x <- wine[, 2:14]  # Features are in columns 2 to 14
y <- wine[, 1]     # Class labels are in column 1

## feature boxplots
boxplot(x, main="wine features")

## class label distributions
plot(y)

# Convert y to a matrix
y_matrix <- as.matrix(y)

# Convert y_matrix to a factor
y_factor <- as.factor(y_matrix)

## feature-class plots (pairs)
featurePlot(x = x, y = y_factor, plot = "pairs")

##(box plots)
featurePlot(x=x, y=y_factor, plot="box")

#distribution plots of each factor
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y_factor, plot="density", scales=scales)

##plot the type distribution for proline
ggplot(wine, aes(x = Proline, y = Type, colour = Type)) +
  geom_point()

## train SVM model - linear kernel
svm.mod0 <- svm(Type ~ ., data = train, kernel = 'linear')

svm.mod0

train.pred <- predict(svm.mod0, train)

cm = as.matrix(table(Actual = train$Type, Predicted = train.pred))

cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted 

recall = diag / rowsums 
precision = diag / colsums
f1 = 2 * precision * recall / (precision + recall) 

data.frame(precision, recall, f1)

##SVM - radial
svm.mod1 <- svm(Type ~ ., data = train, kernel = 'radial')

svm.mod1

train.pred <- predict(svm.mod1, train)

cm = as.matrix(table(Actual = train$Type, Predicted = train.pred))

cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted 

recall = diag / rowsums 
precision = diag / colsums
f1 = 2 * precision * recall / (precision + recall) 

data.frame(precision, recall, f1)

## Tuned SVM - radial
tuned.svm <- tune.svm(
  Type ~ ., 
  data = train, 
  kernel = 'radial', 
  gamma = seq(1 / (2^ncol(wine)), 1, 0.01), # Adjusted gamma range based on the number of features
  cost = 2^seq(-6, 4, 2)                   # Cost parameter range
)

# View the best parameters and performance
summary(tuned.svm)

#View the optimal gamma and cost values - stored in best parameters in tuned.svm
tuned.svm$best.parameters

svm.mod2 <- svm(Type ~ ., data = train, kernel = 'radial', gamma = 0.02006104, cost = 4)

svm.mod2

train.pred <- predict(svm.mod2, train)

cm = as.matrix(table(Actual = train$Type, Predicted = train.pred))

cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted 

recall = diag / rowsums 
precision = diag / colsums
f1 = 2 * precision * recall / (precision + recall) 

data.frame(precision, recall, f1)

######KNN Model############3

normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x))) }

wine[2:14] <- as.data.frame(lapply(wine[2:14], normalize))

summary(wine)

sqrt(177)
k = 13

KNNpred <- knn(train = train[2:14], test = test[2:14], cl = train$Type, k = k)

contingency.table <- table(KNNpred,test$Type)
contingency.table

contingency.matrix = as.matrix(contingency.table)

sum(diag(contingency.matrix))/length(test$Type)

accuracy <- c()
ks <- c(9,11,13,15,17)

for (k in ks) {
  KNNpred <- knn(train = train[2:14], test = test[2:14], cl = train$Type, k = k)
  
  contingency.table <- table(KNNpred, test$Type)  
  contingency.matrix = as.matrix(contingency.table) 
  
  accuracy_k <- sum(diag(contingency.matrix)) / sum(contingency.matrix)  
  accuracy <- c(accuracy, accuracy_k) 
}

plot(ks,accuracy,type = "b")
print(accuracy)

k = 11

KNNpred <- knn(train = train[2:14], test = test[2:14], cl = train$Type, k = k)

contingency.table <- table(KNNpred,test$Type)
contingency.table



############Part 2 - SVM Regression

#SVM Regression model to predict price based off square footage and plot input vs predicted

# remove all columns except for price and square footage
NY_House_Dataset <- NY_House_Dataset[, c(3, 6)]

#remove rows from both columns that have values more than 2 standard deviations away from the median
NY_House_Dataset <- NY_House_Dataset[abs(NY_House_Dataset$PRICE - median(NY_House_Dataset$PRICE, na.rm = TRUE)) <= 2 * sd(NY_House_Dataset$PRICE, na.rm = TRUE) &
                                       abs(NY_House_Dataset$PROPERTYSQFT - median(NY_House_Dataset$PROPERTYSQFT, na.rm = TRUE)) <= 2 * sd(NY_House_Dataset$PROPERTYSQFT, na.rm = TRUE), ]

# Features (Square Footage)
x <- NY_House_Dataset[, "PROPERTYSQFT", drop = FALSE]  # Retain as a data frame with drop = FALSE

# Class label (Price)
y <- NY_House_Dataset[, "PRICE"]

## feature boxplots
boxplot(x, main="sq footage features")

options(scipen = 999)  # Reduce the tendency to use scientific notation
plot(y)  # Plot the data
#options(scipen = 0)    # Reset to default after the plot, if desired

# Convert y to a matrix
y_matrix <- as.matrix(y)

# Convert y_matrix to a factor
y_factor <- as.factor(y_matrix)

## feature-class plots
featurePlot(x=x, y=y_factor, plot="box")

ggplot(NY_House_Dataset, aes(x = PROPERTYSQFT, y = PRICE, colour = PRICE)) +
  geom_point()

# ## split train/test
train.indexes <- sample(4707,0.7*4707)

train <- NY_House_Dataset[train.indexes,]
test <- NY_House_Dataset[-train.indexes,]

#SVM Model - Linear Regression 
## train SVM model - linear kernel
svm.modny <- svm(PRICE ~ ., data = train, kernel = 'linear')

svm.modny

test.pred <- predict(svm.modny, test)

## err = predicted - real
err <- test.pred - test$PRICE

# Plot predicted vs. actual values
ggplot(data.frame(Actual = test$PRICE, Predicted = test.pred), aes(x = Actual, y = Predicted)) +
  geom_point(color = "red") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Predicted vs. Actual Prices", x = "Actual Prices", y = "Predicted Prices")

## MAE
abs.err <- abs(err)
mean.abs.err <- mean(abs.err)

## MSE (more sensitive to outliers than other two methods)
sq.err <- err^2
mean.sq.err <- mean(sq.err)

## RMSE
sq.err <- err^2
mean.sq.err <- mean(sq.err)
root.mean.sq.err <- sqrt(mean.sq.err)

## Tuned SVM - linear regression
tuned.svm <- tune.svm(
  PRICE ~ ., 
  data = train, 
  kernel = 'radial', 
  gamma = seq(1 / (2^ncol(NY_House_Dataset)), 1, 0.01), # Adjusted gamma range based on the number of features
  cost = 2^seq(-6, 4, 2),
  tune.control(cross = 1)
)


# View the best parameters and performance
summary(tuned.svm)

#View the optimal gamma and cost values - stored in best parameters in tuned.svm
tuned.svm$best.parameters

svm.mod2 <- svm(Type ~ ., data = train, kernel = 'radial', gamma = 0.02006104, cost = 4)

svm.mod2

test.pred2 <- predict(svm.mod2, test)

## err = predicted - real
err2 <- test.pred2 - test$PRICE

# Plot predicted vs. actual values
ggplot(data.frame(Actual = test$PRICE, Predicted = test.pred2), aes(x = Actual, y = Predicted)) +
  geom_point(color = "red") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Predicted vs. Actual Prices", x = "Actual Prices", y = "Predicted Prices")

## MAE
abs.err2 <- abs(err2)
mean.abs.err2 <- mean(abs.err2)

## MSE (more sensitive to outliers than other two methods)
sq.err2 <- err2^2
mean.sq.err2 <- mean(sq.err2)

## RMSE
sq.err2 <- err2^2
mean.sq.err2 <- mean(sq.err2)
root.mean.sq.err2 <- sqrt(mean.sq.err2)

##Linear model to predict price based off square footage

# Inspect the structure of the dataset
str(NY_House_Dataset)

# Split the dataset into training and testing sets
set.seed(123)  # For reproducibility
train.indexes <- sample(nrow(NY_House_Dataset), 0.7 * nrow(NY_House_Dataset))

# Create training and testing datasets
train <- NY_House_Dataset[train.indexes, ]
test <- NY_House_Dataset[-train.indexes, ]

# Convert columns to log base 10 form in both training and testing datasets
train$PRICE <- log10(train$PRICE)
train$PROPERTYSQFT <- log10(train$PROPERTYSQFT)

test$PRICE <- log10(test$PRICE)
test$PROPERTYSQFT <- log10(test$PROPERTYSQFT)

# Train a linear model to predict PRICE based on PROPERTYSQFT using the training data
linear_model_NY <- lm(PRICE ~ PROPERTYSQFT, data = train)

# Summary of the linear model
summary(linear_model_NY)

# Predict on the testing data
test$Predicted_PRICE <- predict(linear_model_NY, newdata = test)

# Plot predicted price vs. actual price for the testing dataset
ggplot(test, aes(x = PRICE, y = Predicted_PRICE)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Predicted vs. Actual Prices (Linear Model)",
    x = "Actual Price (log10)",
    y = "Predicted Price (log10)"
  )







