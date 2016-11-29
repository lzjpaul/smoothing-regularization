# logic bug: set regularization to 0 and see the scale for parameters
#
#
# n_folds = 5
# python Example-iris-smoothing.py /data1/zhaojing/regularization/uci-dataset/car_evaluation/car.categorical.data 1 1
# the first 1 is label column, the second 1 is scale or not
from huber_svm import HuberSVC
from smoothing_regularization import Smoothing_Regularization

import pandas
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import ElasticNet, Lasso, RidgeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.preprocessing import scale
from svmlightDataLoader import svmlightclassificationDataLoader

import warnings
import sys
warnings.filterwarnings("ignore")

# data = load_iris()

# # X = data['data']
# X = scale(data['data'])
# y = data['target']
#

labelcol = int(sys.argv[2]) 
X, y = svmlightclassificationDataLoader(sys.argv[1])
# '/data/regularization/car_evaluation/car.categorical.data')
# /data/regularization/Audiology/audio_data/audiology.standardized.traintestcategorical.data
print "using data loader"

# debug: using scale
if int(sys.argv[3]) == 1:
    X = scale(X)
    print "using scale"

print "X.shape = \n", X.shape
print "y.shape = \n", y.shape


idx = np.random.permutation(X.shape[0])
X = X[idx]
y = y[idx]

lasso = OneVsRestClassifier(Lasso())
param_lasso = {'estimator__alpha': [100, 10, 1, 0.1, 1e-2, 1e-3]}

elastic = OneVsRestClassifier(ElasticNet())
param_elastic = {'estimator__alpha': [100, 10, 1, 0.1, 1e-2, 1e-3], 
                 'estimator__l1_ratio': np.linspace(0.01, 0.99, 5)}

ridge = RidgeClassifier(solver='lsqr')
param_ridge = {'alpha': [100, 10, 1, 0.1, 1e-2, 1e-3]}

huber = OneVsRestClassifier(HuberSVC())
param_huber = {'estimator__C': [100, 10, 1, 0.1, 1e-2, 1e-3],
              'estimator__lambd': [100, 10, 1, 0.1, 1e-2, 1e-3], 
              'estimator__mu': [100, 10, 1, 0.1, 1e-2, 1e-3]}

noregulasso = OneVsRestClassifier(Lasso())
param_noregulasso = {'estimator__alpha': [0]}

noreguelastic = OneVsRestClassifier(ElasticNet())
param_noreguelastic = {'estimator__alpha': [0], 
                 'estimator__l1_ratio': np.linspace(0.01, 0.99, 5)}

noreguridge = RidgeClassifier(solver='lsqr')
param_noreguridge = {'alpha': [0]}

noregu = OneVsRestClassifier(Smoothing_Regularization())
param_noregu = {'estimator__C': [100, 10, 1, 0.1, 1e-2, 1e-3],
                'estimator__lambd': [0],
                'estimator__alpha': [10000, 1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]}

smoothing = OneVsRestClassifier(Smoothing_Regularization())
param_smoothing = {'estimator__C': [100, 10, 1, 0.1, 1e-2, 1e-3],
                   'estimator__lambd': [100, 10, 1, 0.1, 1e-2, 1e-3]
                   #'estimator__alpha': [10000, 1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]
                  }

n_folds = 5
param_folds = 3
scoring = 'accuracy'

result_df = pandas.DataFrame()
for i, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
    for clf_name, clf, param_grid in [('Smoothing_Regularization', smoothing, param_smoothing),
                                      ('noregu', noregu, param_noregu),
                                      ('Lasso', lasso, param_lasso),
                                      ('ElasticNet', elastic, param_elastic),
                                      ('Ridge', ridge, param_ridge),
                                      ('noregulasso', noregulasso, param_noregulasso),
                                      ('noreguelastic', noreguelastic, param_noreguelastic), 
                                      ('noreguridge', noreguridge, param_noreguridge)
                                      #('HuberSVC', huber, param_huber)
                                      #('Lasso', lasso, param_lasso)
                                      ]:
        print "clf_name: \n", clf_name
        gs = GridSearchCV(clf, param_grid, scoring=scoring, cv=param_folds, n_jobs=-1)
        gs.fit(X[train_index], y[train_index])
        best_clf = gs.best_estimator_
        
        score = accuracy_score(y[test_index], best_clf.predict(X[test_index]))
        result_df.loc[i, clf_name] = score
        print 'coeficient:', best_clf.coef_, '\n best params:', gs.best_params_, '\n best score', gs.best_score_

print "result shows: \n"
result_df.loc['Mean'] = result_df.mean()
pandas.options.display.float_format = '{:,.3f}'.format
result_df
print result_df.values