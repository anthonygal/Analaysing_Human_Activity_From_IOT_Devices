library(readr)
library(MASS)
library(caTools)

set.seed(100)

# CV function for different model 

cross_validation <-
  function(model,
           data,
           hyperparameters = c(),
           n_folds = 10) {
    n <- nrow(data)
    folds <- sample(1:n_folds, n, replace = TRUE)
    mean <- 0
    errors <- rep(0, n_folds)
    for (k in (1:n_folds)) {
      if (model == 'adl') {
        model.fit <-
          lda(as.factor(class) ~ ., data = data[folds != k,])
        pred <- predict(model.fit, data[folds == k,])
        errors[k] <-
          sum(data$class[folds == k] != pred$class) / length(pred$class)
        mean <- mean + (errors[k] * length(pred$class))
      } else if (model == 'adq') {
        model.fit <-
          qda(as.factor(class) ~ ., data = data[folds != k,])
        pred <- predict(model.fit, data[folds == k,])
        errors[k] <-
          sum(data$class[folds == k] != pred$class) / length(pred$class)
        mean <- mean + (errors[k] * length(pred$class))
      } else if (model == 'svc') {
        kernel <- hyperparameters$kernel
        kpar <- hyperparameters$kpar
        C <- hyperparameters$C
        train <- data[folds != k, ]
        model.fit <-
          ksvm(
            class ~ .,
            data = train,
            type = "C-svc",
            kernel = kernel,
            kpar = kpar,
            C = C,
            cross = 0
          )
        test_x <- data[folds == k, -which(names(data) == "class")]
        test_y <- as.factor(data[folds == k,]$class)
        model.pred <- predict(model.fit, test_x)
        errors[k] <- sum(test_y != model.pred) / length(model.pred)
        mean <- mean + (errors[k] * length(model.pred))
      } else if (model == 'rf') {
        train <- data[folds != k, ]
        test_x <- data[folds == k, -which(names(data) == "class")]
        test_y <- as.factor(data[folds == k,]$class)
        if ("mtry" %in% attributes(hyperparameters)$names) {
          mtry <- hyperparameters$mtry
        } else{
          mtry <- floor(sqrt(ncol(test_x)))
        }
        ntree <- hyperparameters$ntree
        rf <-
          randomForest(class ~ .,
                       data = train,
                       mtry = mtry,
                       ntree = ntree)
        model.pred <-
          predict(rf, newdata = test_x, type = "response")
        errors[k] <- sum(test_y != model.pred) / length(model.pred)
        mean <- mean + (errors[k] * length(model.pred))
      } else{
        stop("Modèle inconnu \n")
      }
    }
    mean <- mean / n
    sd <-
      sqrt((1 / (n_folds - 1)) * sum((errors - rep(mean, n_folds)) ^ 2))
    return(c(mean, sd))
  }


smartphone <- read_csv("donnees/smartphone.csv") # -> contextual data

nrow(smartphone) # 1.528.218
ncol(smartphone) # 4

summary(smartphone)

colSums(is.na(smartphone)) # pas de NA

sapply(smartphone, function(x) length(unique(x))) # nombre de valeurs uniques pour chaque parametres

smartphone$source <- factor(smartphone$source)


# 17 sources 

unique(smartphone[smartphone$source == 'activity',]$values) 
# activity label : STILL, ON_FOOT, WALKING, IN_VEHICLE, ON_BICYCLE, RUNNING, UNKNOWN


unique(smartphone$source)
# VALUES :
# - Light //
# - Audio //
# - Battery //
# - Orientation //

# - Step_Detector.
# - Step_Counter.

# - Magnetometer :
# - Gravity : 
# - Gyroscope : 
# - Pressure : 

# - Linear_Acceleration : the acceleration force in ‘meters per second’ that is applied 
# to a device on all three physical axes (x, y, and z), excluding the force of gravity.

# - Accelerometer : the acceleration force in ‘meters per second’ that is applied 
# to a device on all three physical axes (x, y, and z), including the force of gravity.

# - Wifi : List of Wifi access points (like [7207, 5500, 3564, 3544])

# - Bluetooth : List of BT devices accessible ([815])

# - Proximity
# - rotationVector : 

nrow(smartphone[smartphone$source == 'activity',]) # -> 16.461 obs

indices = sample.split(smartphone$source, SplitRatio = 0.7)

train = smartphone[indices,]
test = smartphone[!(indices),]