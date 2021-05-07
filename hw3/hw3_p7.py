from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import mean_absolute_error as MAE


def dataset(test_size=.3):
    digits = load_digits()
    # print(dir(digits))
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    return X, y, X_train, X_test, y_train, y_test


def show_img():
    digits = load_digits()
    for i in digits.target_names:
        plt.imshow(digits.images[i], cmap=plt.cm.gray_r)
        plt.axis("off")
        plt.show()


def confusion_matrix(y_test, predictions, score):
    cm = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Target label')
    plt.xlabel('Predicted label')
    title = f'Accuracy Score: {score:.5f}'
    plt.title(title, size=12)
    plt.show()


def mae(y_test, predictions):
    mae = MAE(y_test, predictions)
    print('absolute error:', mae * len(y_test))


def knn(n_neighbors, train_size=None):

    _, _, X_train, X_test, y_train, y_test = dataset()
    if train_size:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=train_size)

    knn = KNN(n_neighbors)
    knn.fit(X_train, y_train)

    if train_size:
        print('with train size:',train_size)
    print('knn score:%f  with K=%i neighbors' % (knn.score(X_test, y_test), n_neighbors))
    mae(y_test, knn.predict(X_test))
    confusion_matrix(y_test, knn.predict(X_test), knn.score(X_test, y_test))


def lda(train_size=None):
    _, _, X_train, X_test, y_train, y_test = dataset()
    if train_size:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=train_size)

    lda = LDA()
    lda.fit(X_train, y_train)
    mae(y_test, lda.predict(X_test))
    confusion_matrix(y_test, lda.predict(X_test), lda.score(X_test, y_test))


def qda(train_size=None):
    _, _, X_train, X_test, y_train, y_test = dataset()
    if train_size:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=train_size)

    qda = QDA()
    qda.fit(X_train, y_train)
    mae(y_test, qda.predict(X_test))
    confusion_matrix(y_test, qda.predict(X_test), qda.score(X_test, y_test))


def logistic_reg(train_size=None):
    _, _, X_train, X_test, y_train, y_test = dataset()
    if train_size:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=train_size)

    lreg = LogisticRegression(solver='newton-cg', multi_class='auto', max_iter=100)
    lreg.fit(X_train, y_train)
    predictions = lreg.predict(X_test)
    score = lreg.score(X_test, y_test)
    mae(y_test, predictions)
    confusion_matrix(y_test, predictions, score)


### section_1
show_img()

#### section_2 & 3
knn(5)
knn(10)
knn(30)
lda()
qda()
logistic_reg()

#### section_4

knn(5, train_size=.1)
knn(10, train_size=.1)
knn(30, train_size=.1)
lda(train_size=.1)
qda(train_size=.1)
logistic_reg(train_size=.1)

