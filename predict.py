from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV

np.random.seed(0)
data = pd.read_csv('D:\\Coding\\data_science\\Upgrad\\bank-additional-full.csv',sep=';')

data=data.drop(['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','previous','contact','housing','day_of_week','loan','duration'],axis=1)
y=data['y']
X=data.drop(['y'],axis=1)
#data=data.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
#preprocessing
y=pd.get_dummies(y)
X=pd.get_dummies(X)
'''
print (data['y'].sort_values(ascending=False), '\n')
print(data)
ax = sns.heatmap(data)
plt.show()
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
'''
param_grid = { 
    'n_estimators': list(range(2,20)),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : list(range(2,20)),
    'criterion' :['gini', 'entropy']
}
clf=RandomForestClassifier()
CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)
'''

clf=RandomForestClassifier(n_estimators=13,max_depth=7)
clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores.mean())

