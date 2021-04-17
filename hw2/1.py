import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matplotlib import pylab as plt


X = np.array([0.5, 1.1, 2., 2.9]).reshape(-1, 1)
y = np.array([2, 3.3, 5.7, 7.8])
reg = LinearRegression().fit(X, y)
predictions = reg.predict(X)

print(reg.intercept_, reg.coef_)
print('MSE:', metrics.mean_squared_error(y, predictions))
print('regression line: y = %fx + %i' % (reg.coef_, reg.intercept_))
plt.scatter(X, y, color='blue')
plt.plot(X, predictions, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

### P_1 part b
# x[:, 1] = [ null, .9, 2, null]
mean = (.9+2)/2
x = np.append(X, np.array([mean, 0.9, 2, mean]).reshape(-1, 1), 1)
reg = LinearRegression().fit(x, y)
predictions = reg.predict(x)

print(reg.intercept_, reg.coef_)
print('MSE:', metrics.mean_squared_error(y, predictions))
print('regression plane: y = %fx + %gz + %i' % (reg.coef_[0], reg.coef_[1], reg.intercept_))

### P_1 part e and d
'''we fit the model on the second feature as labels
I added the y array to the known x_1 feature, together they form input data for the regression
and x_2 features which are missing is our labels array'''

x_new = np.append(x[:, 0].reshape(4, 1), y.reshape(4, 1), 1)
reg = LinearRegression().fit(x_new.reshape(4, 2), x[:, 1])
predictions = reg.predict(x_new.reshape(4, 2))
# adding the missing values to the feature array from the predicted data
x[0, 1] = predictions[0]
x[3, 1] = predictions[3]
print(x)

reg = LinearRegression().fit(x, y)
predictions = reg.predict(x)
print(reg.intercept_, reg.coef_)
print('MSE:', metrics.mean_squared_error(y, predictions))
print('regression plane: y = %fx + %gz + %i' % (reg.coef_[0], reg.coef_[1], reg.intercept_))