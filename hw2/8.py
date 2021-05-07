import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def dataset():
    ozone_dataset = pd.read_csv('LAozone.data', sep=',')
    X = ozone_dataset[['vh', 'wind', 'humidity', 'temp', 'ibh', 'dpg', 'ibt', 'vis', 'doy']]
    y = ozone_dataset['ozone']
    return X, y


def forward_selection(data, target, significance_level=5):
    """

    :param data: pandas dataframe of input data
    :param target: pandas dataframe of input data's corresponding target
    :param significance_level: compared to p_values, higher p_value means less important feature.
    therefore, the higher the significance level, the more features will be listed
    :return: the features in order of their importance
    """
    initial_features = data.columns.tolist()
    best_features = []
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features, dtype=float)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features


def forward_selection_regression(data, target, k_features=3):
    """

    :param data: pandas dataframe of input data
    :param target: pandas dataframe of input data's corresponding target
    :param k_features: number of desired features to fit the regression upon,
    features are chosen based on their importance
    :return: prints out the mean squared error and regression coefficients
    """

    reg = LinearRegression()
    sfs = SFS(reg,
              k_features,
              forward=True,
              floating=False,
              verbose=0,
              scoring='r2',
              cv=5)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.3)
    sfs = sfs.fit(X_train, y_train)
    X_train_sfs = sfs.transform(X_train)
    X_test_sfs = sfs.transform(X_test)
    reg = reg.fit(X_train_sfs, y_train)

    print('estimated coefficients for the linear regression:', reg.coef_)
    print('interception coefficient b_0:', reg.intercept_)
    print('MSE_train:', metrics.mean_squared_error(y_train, reg.predict(X_train_sfs)))
    print('MSE_test:', metrics.mean_squared_error(y_test, reg.predict(X_test_sfs)))


def regression(X_train, y_train, reg_type='linear', alpha=.0):
    """
    :param X_train: pandas dataframe of training data
    :param y_train: pandas dataframe of training data's labels
    :param reg_type: linear, lasso, ridge
    :param alpha: shrinkage coefficient
    :return: returns the regression model
    """
    global reg
    if reg_type == 'linear':
        reg = LinearRegression().fit(X_train, y_train)
    elif reg_type == 'lasso':
        reg = linear_model.Lasso(alpha, normalize=True, max_iter=10000)
        reg.fit(X_train, y_train)
    elif reg_type == 'ridge':
        reg = linear_model.Ridge(alpha)
        reg.fit(X_train, y_train)
    else:
        print('please specify the regression method')
    return reg


def print_mse(X, y, reg_type='linear', alpha=0):
    """
        :param alpha: shrinkage coefficient
        :param reg_type: linear, lasso, ridge
        :param X: pandas dataframe of input data
        :param y: pandas dataframe of labels
        :return: prints out the mean squared error after removing one feature
    """

    L = np.arange(len(list(X.columns)))
    MSE_train = np.zeros(len(list(X.columns)))
    MSE_test = np.zeros(len(list(X.columns)))
    var_noise = []

    for i in range(len(list(X.columns))):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.97)
        reg = regression(X_train, y_train, reg_type, alpha)
        MSE_train[i] = metrics.mean_squared_error(y_train, reg.predict(X_train))
        MSE_test[i] = metrics.mean_squared_error(y_test, reg.predict(X_test))
        rss = np.sum(np.square(y_train - reg.predict(X_train)))
        var_noise.append(rss / (X.shape[0] - len(list(X.columns)) - 1))

        if len(list(X.columns)) == 1: break
        X = X.drop(columns=X.columns[-1])

    plt.plot(L, MSE_train, color="orange", label="MSE_train")
    plt.plot(L, MSE_test, color="purple", label="MSE_test")
    plt.plot(L, var_noise, color="grey", label="var_noise")
    plt.xlabel('number of deleted features')
    plt.ylabel('$\\lambda$:' + str(alpha))
    plt.title('regression method:' + reg_type)
    plt.legend()
    plt.show()


X, y = dataset()
#### Problem_8 section_a
# print(forward_selection(X, y))

#### Problem_8 section_b
# forward_selection_regression(X, y, k_features=3)

#### Problem_8 section_c
# print_mse(X, y)

#### Problem_8 section_d
# lambdas = [0.001, 0.01, 0.1, 0.5, 1, 2, 10]
# for i in lambdas:
#     print_mse(X, y, 'lasso', alpha=i)
#     print_mse(X, y, 'ridge', alpha=i)
