import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('perceptron-train.csv', header=None)
df_test = pd.read_csv('perceptron-test.csv', header=None)

y_train, X_train = df_train.values[:, 0], df_train.values[:, 1:]
y_test, X_test = df_test.values[:, 0], df_test.values[:, 1:]

clf = Perceptron(random_state=241).fit(X_train, y_train)
print clf.score(X_test, y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_scaled = Perceptron(random_state=241).fit(X_train_scaled, y_train)
print clf_scaled.score(X_test_scaled, y_test)