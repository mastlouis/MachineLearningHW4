from cvxopt import solvers, matrix
import numpy as np
import sklearn.svm

class SVM4342 ():
    def __init__ (self):
        pass

    # Expects each *row* to be an m-dimensional row vector. X should
    # contain n rows, where n is the number of examples.
    # y should correspondingly be an n-vector of labels (-1 or +1).
    def fit (self, X, y):
        # TODO change these -- they should be matrices or vectors
        Xbiased = append_ones(X)
        yprime = np.tile(y * -1, (Xbiased.shape[1], 1)).T
        itilde = np.eye(Xbiased.shape[1])
        itilde[-1,-1] = 0
        G = yprime * Xbiased
        P = itilde
        q = np.zeros(Xbiased.shape[1])
        h = np.repeat(-1, y.shape[0])

        # Solve -- if the variables above are defined correctly, you can call this as-is:
        sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))

        # Fetch the learned hyperplane and bias parameters out of sol['x']
        self.w = np.atleast_2d(np.array(sol['x'].H[:-1])).T
        self.b = sol['x'].H[-1]

    # Given a 2-D matrix of examples X, output a vector of predicted class labels
    def predict (self, x):
        yhat = np.zeros(x.shape[0])
        bools = x.dot(self.w[0, :]) >= 0 - self.b
        yhat[bools] = 1
        yhat[yhat == 0] = -1
        return yhat

def append_ones(array):
    return np.hstack((array, np.ones((array.shape[0], 1))))

def test1 ():
    # Set up toy problem
    X = np.array([ [1,1], [2,1], [1,2], [2,3], [1,4], [2,4] ])
    y = np.array([-1,-1,-1,1,1,1])

    # Train your model
    svm4342 = SVM4342()
    svm4342.fit(X, y)
    print(svm4342.w, svm4342.b)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard-margin
    svm.fit(X, y)
    print(svm.coef_, svm.intercept_)

    acc = np.mean(svm4342.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

def test2 (seed):
    np.random.seed(seed)

    # Generate random data
    X = np.random.rand(20,3)
    # Generate random labels based on a random "ground-truth" hyperplane
    while True:
        w = np.random.rand(3)
        y = 2*(X.dot(w) > 0.5) - 1
        # Keep generating ground-truth hyperplanes until we find one
        # that results in 2 classes
        if len(np.unique(y)) > 1:
            break

    svm4342 = SVM4342()
    svm4342.fit(X, y)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard margin
    svm.fit(X, y)
    diff = np.linalg.norm(svm.coef_ - svm4342.w) + np.abs(svm.intercept_ - svm4342.b)
    print(diff)

    acc = np.mean(svm4342.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

    if acc == 1 and diff < 1e-1:
        print("Passed")

if __name__ == "__main__": 
    test1()
    for seed in range(5):
        test2(seed)
