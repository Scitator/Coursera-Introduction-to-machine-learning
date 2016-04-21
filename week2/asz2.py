import pandas as pd
import operator
from sklearn.cross_validation import KFold
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn import cross_validation

boston = load_boston()

data_scaled = scale(boston.data)

kf = KFold(boston.data.shape[0], n_folds=5, shuffle=True, random_state=42)

step1_res = {}

for train_index, test_index in kf:
	X_train, X_test = boston.data[train_index], boston.data[test_index]
	y_train, y_test = boston.target[train_index], boston.target[test_index]
	for x in np.linspace(1.0, 10.0, num=200):
		neigh = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(data_scaled, boston.target) 
		step1_res[x] = max(cross_validation.cross_val_score(neigh, boston.data, boston.target.ravel(), cv=5, scoring='mean_squared_error'))

step1_res_sorted = sorted(step1_res.items(), key=operator.itemgetter(1), reverse=True)

print step1_res_sorted[0:3]