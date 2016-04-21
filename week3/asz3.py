import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data-logistic.csv', header=None)
y = df.values[:, 0]
X = df.values[:, 1:]
w1 = 0
w2 = 0
eps = 0.00001
k = 0.1
C = 10

for  i in range(10000):
	w1_new_sum = 0
	w2_new_sum = 0
	for l in range(y.size):
		w1_new_sum = w1_new_sum +  y[l]*X[l,0]*(1 - 1/(1+ np.exp(-y[l]*(w1*X[l,0] + w2*X[l,1]))))
		w2_new_sum = w2_new_sum +  y[l]*X[l,1]*(1 - 1/(1+ np.exp(-y[l]*(w1*X[l,0] + w2*X[l,1])))) 
	w1_new = w1 + (k/y.size)*w1_new_sum - k*C*w1
	w2_new = w2 + (k/y.size)*w2_new_sum - k*C*w2
	r = np.sqrt((w1 - w1_new)**2 + (w1 - w1_new)**2)
	w1 = w1_new
	w2 = w2_new
	if r < eps:
		break
print w1
print w2

yRes = []
for i in range(y.size):
	yRes.append(w1*X[i,0] + w2*X[i,1])
print roc_auc_score(y,yRes)