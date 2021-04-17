from sklearn import linear_model
# import numpy as np
#
# X = np.array([-2, -1, -1, -1, 0, 1, 2, 2]).reshape(-1, 1)
# y = np.array([35, 40, 36, 38, 40, 43, 45, 43])
# clf = linear_model.Lasso(alpha=0.1)
# clf.fit(X, y)
# print('regression line: y = %fx + %i' % (clf.coef_, clf.intercept_))
#
# clf = linear_model.Lasso(alpha=1)
# clf.fit(X, y)
# print('regression line: y = %fx + %i' % (clf.coef_, clf.intercept_))
#
# clf = linear_model.Lasso(alpha=3)
# clf.fit(X, y)
# print('regression line: y = %fx + %i' % (clf.coef_, clf.intercept_))

#############

import numpy as np
import matplotlib.pyplot as plt


# Lasso Regression
class LassoRegression:
    def __init__(self, learning_rate, iterations, l1_penality):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penality = l1_penality

    # Function for model training
    def fit(self, X, Y):
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape
        # weight initialization
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        # gradient descent learning
        for i in range(self.iterations):
            self.update_weights()
        return self

    # update weights using gradient descent
    def update_weights(self):
        Y_pred = self.predict(self.X)
        # calculate gradients
        dW = np.zeros(self.n)
        for j in range(self.n):
            if self.W[j] > 0:
                dW[j] = (- (2 * (self.X[:, j]).dot(self.Y - Y_pred)) + self.l1_penality) / self.m
            else:
                dW[j] = (- (2 * (self.X[:, j]).dot(self.Y - Y_pred)) - self.l1_penality) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m
        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def predict(self, X):
        return X.dot(self.W) + self.b

    def fitted_line(self, x, y_pred):
        slope = (y_pred[-1] - y_pred[0]) / (x[-1] - x[0])
        print('y=%f( x + %s )' % (slope, y_pred[0] - slope * x[0]))


# Importing dataset
X = np.array([-2, -1, -1, -1, 0, 1, 2, 2]).reshape(-1, 1)
y = np.array([35, 40, 36, 38, 40, 43, 45, 43])
# Model training
lambads = [0.01, 0.1, 0.5, 1, 5, 10, 20, 30, 40, 50]
for i in lambads:
    model = LassoRegression(iterations=1000, learning_rate=0.01, l1_penality=i)
    model.fit(X, y)
    y_pred = model.predict(X)
    model.fitted_line(X, y_pred)
    # Visualization on test set
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='orange')
    plt.title('$\\lambda$:%f' %i)
    plt.show()
