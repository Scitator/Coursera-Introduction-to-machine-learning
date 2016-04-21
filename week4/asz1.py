import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack


data_train = pd.read_csv('salary-train.csv', header=0)
data_test = pd.read_csv('salary-test-mini.csv', header=0)

data_train.update(data_train['FullDescription'].str.lower().apply(lambda x: re.sub('[^a-z0-9]', ' ', x)))
#data_test.update(data_test['FullDescription'].str.lower().apply(lambda x: re.sub('[^a-z0-9]', ' ', x)))


Tfidf = TfidfVectorizer(min_df = 5.0/len(data_train['FullDescription']))
X_train = Tfidf.fit_transform(data_train['FullDescription'] )
X_test = Tfidf.fit(data_test['FullDescription'] )

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)


enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

df_train_clean = hstack([X_train, X_train_categ])
#df_test_clean = hstack([X_test, X_test_categ])
print X_train
print len(X_train)
print ""
print X_train_categ
print len(X_train_categ)
print ""
print df_train_clean
print len(df_train_clean)
print ""