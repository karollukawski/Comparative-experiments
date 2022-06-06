import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from numpy.random import MT19937
from numpy.random import RandomState
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neighbors import DistanceMetric, KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.base import clone
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances, accuracy_score
from tabulate import tabulate


classifiers = [
    tree.DecisionTreeClassifier(),
    KNeighborsClassifier(),
    SVC (),
]


X, y = make_classification(n_samples=500,n_informative=5,n_features=30)

mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, X.shape[1])
X = X * s

rskf = RepeatedStratifiedKFold(n_splits=5,n_repeats=5)
rskf.get_n_splits(X, y)


average =[]
deviation =[]

for cf_cnt, cf in enumerate(classifiers):
    pkt = []
    for train_index, test_index in rskf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clone(cf)
        clf.fit(X_train,y_train)
        pred = clf.predict(X_test)
        pkt.append(accuracy_score(y_test, pred))

    average.append(np.mean(pkt))
    deviation.append(np.std(pkt))


table = [['','CART','kNN','SVC'],
         ["Average",(format(average[0],".3f")),(format(average[1],".3f")),(format(average[2],".3f"))],
         ["Deviation",(format(deviation[0],".2f")),(format(deviation[1],".2f")),(format(deviation[2],".2f"))]]
print(tabulate(table))