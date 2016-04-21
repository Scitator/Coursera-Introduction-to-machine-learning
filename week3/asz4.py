import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

df = pd.read_csv('classification.csv', header=0)
y = df.values[:, 0]
X = df.values[:, 1:]
y = map(int, y)
X = map(int, X)

TP = 0
FP = 0
FN = 0
TN = 0

for i in range(len(y)):
	if y[i] == 1 and X[i] == 1:
		TP = TP + 1
	if y[i] == 0 and X[i] == 1:
		FP = FP + 1
	if y[i] == 1 and X[i] == 0:
		FN = FN + 1
	if y[i] == 0 and X[i] == 0:
		TN = TN + 1

print "Question 1:"
print TP # 43
print FP # 34
print FN # 59
print TN # 64
print ""

accuracy = float(TP + TN) / (TP + FP + FN + TN)
precision = float(TP) / (TP + FP)
recall = float(TP) / (TP + FN)
F = 2* precision * recall / (precision + recall)

print "Question 2:"
print accuracy	# 0.54
print precision	# 0.56
print recall	# 0.42
print F 		# 0.48
print ""

df = pd.read_csv('scores.csv', header=0)

print roc_auc_score(df.values[:, 0], df.values[:, 1])
print roc_auc_score(df.values[:, 0], df.values[:, 2])
print roc_auc_score(df.values[:, 0], df.values[:, 3])
print roc_auc_score(df.values[:, 0], df.values[:, 4])
print "" # score_logreg

for i in xrange(1, 5):
	precision, recall, thresholds = precision_recall_curve(df.values[:, 0], df.values[:, i])
	best_precision = 0.0
	for j in range(len(recall)):
		if recall[j] > 0.7 and precision[j] > best_precision:
			best_precision = precision[j]
	print best_precision
# score_tree