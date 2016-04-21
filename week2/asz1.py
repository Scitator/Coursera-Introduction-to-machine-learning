import pandas as pd
import operator
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import cross_validation
from sklearn.preprocessing import scale

df = pd.read_csv('wineData.csv', header=None)

y = df.values[:, 0]
X = df.values[:, 1:]

step1_res = {}

kf = KFold(df.shape[0], n_folds=5, shuffle=True, random_state=42)

for train_index, test_index in kf:
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	for k in range(1, 20):
		neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train.ravel())
		step1_res[k] = cross_validation.cross_val_score(neigh, X, y.ravel(), cv=5).mean()

step1_res_sorted = sorted(step1_res.items(), key=operator.itemgetter(1), reverse=True)

print step1_res_sorted[0:3]

X_scaled = scale(X)
y_scaled = y

step2_res_scaled = {}

kf2 = KFold(df.shape[0], n_folds=5, shuffle=True, random_state=42)

for train_index, test_index in kf2:
	X_train_scaled, X_test_scaled = X_scaled[train_index], X_scaled[test_index]
	y_train_scaled, y_test_scaled = y_scaled[train_index], y_scaled[test_index]
	for k in range(1, 50):
		neigh_scaled = KNeighborsClassifier(n_neighbors=k).fit(X_train_scaled, y_train_scaled)
		step2_res_scaled[k] = cross_validation.cross_val_score(neigh_scaled, X_scaled, y_scaled, cv=6).mean()

step2_res_scaled_sorted = sorted(step2_res_scaled.items(), key=operator.itemgetter(1), reverse=True)

print step2_res_scaled_sorted[0:20]