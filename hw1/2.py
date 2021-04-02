import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

boston_dataset = pd.read_csv('housing.data', sep=',', names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                                                             'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])
# print(boston_dataset.head())
# boston_dataset.info()
# boston_dataset.describe()

X = boston_dataset[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = boston_dataset['MEDV']

# sns.heatmap(X.cov(), cmap='RdGy')
# plt.show()
#
pd.set_option('display.max_columns', None)
cov = X.cov()
print(cov)
##########

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7)
reg = LinearRegression().fit(X_train, y_train)
predictions = reg.predict(X_test)
print('estimated coefficients for the linear regression:', reg.coef_)
print('interception coefficient b_0 linear regression:', reg.intercept_)

# plt.scatter(y_test, predictions)
# plt.xlabel('Y Test')
# plt.ylabel('Predicted Y')
# plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))