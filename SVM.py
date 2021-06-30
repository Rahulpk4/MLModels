from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC

data = load_breast_cancer()

print (data.data)
print (data.feature_names)
print (data.target_names)

X = data.data
Y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

clf = SVC(kernel='linear',C=3)

clf.fit(x_train,y_train)

print (clf.score(x_test,y_test))
