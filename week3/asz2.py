import pandas as pd
import numpy as np
import operator
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(
    subset='all', 
    categories=['alt.atheism', 'sci.space']
)

Tfidf = TfidfVectorizer()

X = Tfidf.fit_transform(newsgroups.data)
y = newsgroups.target


#grid = {'C': np.power(10.0, np.arange(-5, 6))}
#cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
#clf = SVC(kernel='linear', random_state=241)
#gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
#gs.fit(X, y)

#res = {}
#for a in gs.grid_scores_:
#	res[a.parameters['C']] = a.mean_validation_score

#res_sorted = sorted(res.items(), key=operator.itemgetter(1), reverse=True)
#print res_sorted[0]

# best is 10

clf = SVC(C = 10, kernel='linear', random_state=241).fit(X,y)

coefs = np.argsort(np.absolute(np.asarray(clf.coef_.todense()).reshape(-1)))
print 

for index in coefs[-10:]:
	print Tfidf.get_feature_names()[index]

# atheism,atheists,bible,god,keith,moon,nick,religion,sky,space