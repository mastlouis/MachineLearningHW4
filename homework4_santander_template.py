from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import numpy as np
import pandas

# Linear SVM

# Load data
# train.csv contains 200,000 entries
d = pandas.read_csv('train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:, 2:])  # Features

#half the number of all the examples
half = 100000

# Split into train/test folds

# Ensures that the randoms remain the same everytime
np.random.seed(0)
data = np.arange(X.shape[0])

# Shuffles the ordering of the data
np.random.shuffle(data)

# Creating split data sets
y = y[data]
X = X[data]

# Creating training and testing sets
xTrain = X[:half, :]
xTest = X[half:, :]

yTrain = y[:half, ]
yTest = y[half:, ]

# Apply the SVMs to the test set
classifier = LinearSVC(dual=False)
classifier.fit(xTrain, yTrain)

# yhat1
yhat1 = classifier.decision_function(xTest) # Linear kernel
# yhat1 = classifier.predict(xTest) # Linear kernel

# Compute AUC
auc1 = roc_auc_score(yTest, yhat1)
print('Linear Kernel SVM Accuracy:', auc1)



# Non-linear SVM (polynomial kernel)
# Bootstrap Aggregation (Bagging)

#The amount we want in each subset
subSetNum = 5000

#Our predictions
yhat2 = []

for i in range(0, xTrain.shape[0], subSetNum):
   #Different sets used for training and testing
    train = xTrain[i: (i + subSetNum), : ]
    trainLabels = yTrain[i: (i + subSetNum), ]

    #Testing set, not used yet
    test = xTest[i: (i + subSetNum), :]
    testLabels = yTest[i: (i + subSetNum), ]

    # Apply the SVMs to the test set
    classifier = SVC(kernel='poly', degree=3, gamma='auto') # Non-linear kernel
    classifier.fit(train, trainLabels)
    # yhat2.append(classifier.decision_function(xTest))
    yhat2.append(classifier.predict(xTest))

# yhat1 Calculations
yhat2 = np.asarray(yhat2)
yhat3 = np.sum(yhat2, axis=0)
yhat3 /= yhat2.shape[0]
yhat3[yhat3 > 0.5] = 1
yhat3[yhat3 != 1] = 0

# Compute AUC
auc2 = roc_auc_score(yTest, yhat3)
print('Non-Linear SVM Accuracy:', auc2)
