from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
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
xtr = X[:half, :]
xte = X[half:, :]

ytr = y[:half, ]
yte = y[half:, ]

# Apply the SVMs to the test set
classifier = LinearSVC(dual=False)
classifier.fit(xtr, ytr)

# Predictions
yhat1 = classifier.decision_function(xte) # Linear kernel

# Compute AUC
auc1 = roc_auc_score(yte, yhat1)
print('Linear Kernel SVM Accuracy:', auc1)



# Non-linear SVM (polynomial kernel)


# Apply the SVMs to the test set
# yhat2 = ...  # Non-linear kernel

# Compute AUC
# auc2 = ...



