from sklearn import svm
import numpy as np
from sklearn.datasets import load_iris
import random
import pandas as pd


CLASSES = ['setosa', 'versicolor', 'virginica']
Iris = load_iris()

iris_data = pd.DataFrame(data=np.c_[Iris['data'], Iris['target']], columns=Iris['feature_names'] + ['target'])
iris_data['target'] = iris_data['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

X_data = iris_data.iloc[:, [0]].to_numpy()
Y_data = iris_data.iloc[:, [-1]].to_numpy()
new_Y_data = []

for i in Y_data:
    new_Y_data.append(CLASSES.index(i.item()))
Y = np.array(new_Y_data)
X = X_data
print(X, Y)
"""
X = [[0, 0], [1, 1]]
Y = [0, 1]
"""
clf = svm.SVC(kernel='linear', C=1000)
for epoch in range(0, 1000):
    clf.fit(X, Y)
    data = random.randint(0, 100)
    data = data / 10
    result = clf.predict([[float(data)]])
    print(f"Epoch {epoch} | input: {data} > label: {result.item()} | class name: {CLASSES[result.item()]}")
   


