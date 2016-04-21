import pandas as pd
import numpy as np
from sklearn.svm import SVC

df = pd.read_csv('svm-data.csv', header=None)
y = df.values[:, 0]
X = df.values[:, 1:]

clf = SVC(C = 100000, kernel='linear', random_state=241).fit(X,y)

print clf.support_
print clf.support_vectors_