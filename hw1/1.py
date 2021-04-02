import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ozone_dataset = pd.read_csv('LAozone.data', sep=',')
# print(ozone_dataset.head())
# ozone_dataset.info()
# ozone_dataset.describe()

# ozone_dataset.plot.scatter('ozone', 'humidity')
# plt.show()


# sns.heatmap(ozone_dataset.corr(), cmap='RdGy')
# plt.show()
# sns.pairplot(ozone_dataset, vars=['ozone','vh','wind','humidity','temp','ibh','dpg','ibt','vis','doy'])
# plt.show()

X = ozone_dataset[['vh','wind','humidity','temp','ibh','dpg','ibt','vis','doy']]
y = ozone_dataset['ozone']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
reg = LinearRegression().fit(X_train, y_train)
predictions = reg.predict(X_test)
print('estimated coefficients for the linear regression:', reg.coef_)
print('interception coefficient b_0 linear regression:', reg.intercept_)

plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

####### performing PCA dimension reduction
# Standardizing the features
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=1)
principalComponent = pca.fit_transform(X)
principalComponent = pd.DataFrame(data=principalComponent, columns=['principal component'])

pc_train, pc_test, y_train, y_test = train_test_split(principalComponent, y, test_size=.3)
reg1 = LinearRegression().fit(pc_train, y_train)
predictions1 = reg1.predict(pc_test)

print('estimated coefficients for the linear regression:', reg1.coef_)
print('interception coefficient b_0 linear regression:', reg1.intercept_)

print('MAE:', metrics.mean_absolute_error(y_test, predictions1))
print('MSE:', metrics.mean_squared_error(y_test, predictions1))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions1)))
plt.scatter(pc_test, y_test, color='g')
plt.plot(pc_test, predictions1, color='k')
plt.xlabel('principalComponent')
plt.ylabel('y_test')
plt.show()

####### section d

horizontal_stack = pd.concat([principalComponent, y], axis=1)

median = horizontal_stack['principal component'].median()
df_1 = horizontal_stack[horizontal_stack['principal component'] <= median]
df_2 = horizontal_stack[horizontal_stack['principal component'] >= median]
x_1 = df_1[['principal component']]
y_1 = df_1['ozone']
x_2 = df_2[['principal component']]
y_2 = df_2['ozone']

pc1_train, pc1_test, y1_train, y1_test = train_test_split(x_1, y_1, test_size=.3)
reg21 = LinearRegression().fit(pc1_train, y1_train)
predictions21 = reg21.predict(pc1_test)

print('df_1_MAE:', metrics.mean_absolute_error(y1_test, predictions21))
print('df_1_MSE:', metrics.mean_squared_error(y1_test, predictions21))
print('df_1_RMSE:', np.sqrt(metrics.mean_squared_error(y1_test, predictions21)))

plt.scatter(pc1_test, y1_test, color='g')
plt.plot(pc1_test, predictions21, color='k')
plt.xlabel('dataframe 1 of principalComponent')
plt.ylabel('y_test')
plt.show()
####
pc2_train, pc2_test, y2_train, y2_test = train_test_split(x_2, y_2, test_size=.3)
reg22 = LinearRegression().fit(pc2_train, y2_train)
predictions22 = reg21.predict(pc2_test)

print('df_2_MAE:', metrics.mean_absolute_error(y2_test, predictions22))
print('df_2_MSE:', metrics.mean_squared_error(y2_test, predictions22))
print('df_2_RMSE:', np.sqrt(metrics.mean_squared_error(y2_test, predictions22)))

#
plt.scatter(pc2_test, y2_test, color='g')
plt.plot(pc2_test, predictions22, color='k')
plt.xlabel('dataframe 2 of principalComponent')
plt.ylabel('y_test')
plt.show()