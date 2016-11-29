#11-23 adding parama lambda for tuning parameters
__author__ = "Luo Zhaojing"
__version__ = "0.1.0"

import numpy as np
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
import random
from scipy import sparse

def smoothing_grad_descent(X, y, w, param, C, batch_size, batch_iter):
    # sparse matrix works, random.shuffle
    # shuffle: next time shuffle index will be forgetten (static variable: smoothing_grad_descent.idx)
    if not hasattr(smoothing_grad_descent, "idx"):
        smoothing_grad_descent.idx = np.random.permutation(X.shape[0])
    print "X.shape: ", X.shape
    print "max idx: ", max(smoothing_grad_descent.idx)
    X = X[smoothing_grad_descent.idx]
    y = y[smoothing_grad_descent.idx]
    index = (batch_size * batch_iter) % X.shape[0]
    if (index + batch_size) > X.shape[0]: #new epoch
        batch_X, batch_y = X[0 : batch_size], y[0: batch_size] #fixed shape
        batch_X[0 : (X.shape[0] - index)], batch_y[0 : (X.shape[0] - index)] = X[index : X.shape[0]], y[index : X.shape[0]]
        batch_X[(X.shape[0] - index) : batch_size], batch_y[(X.shape[0] - index) : batch_size] = X[0 : (index + batch_size - X.shape[0])], y[0 : (index + batch_size - X.shape[0])]
        # trt print the idx
        smoothing_grad_descent.idx = np.random.permutation(X.shape[0])
    else: # need else here
        batch_X, batch_y = X[index : (index + batch_size)], y[index : (index + batch_size)]
    
    if sparse.issparse(batch_X): batch_X = batch_X.toarray()

    grad =  param * (np.exp(w) - 1) / (np.exp(w) + 1) # log(1+e^(-w)) + log(1+e^(w))
    # print 'grad.shape: ', grad.shape
    f1 = np.exp(-batch_y * np.dot(w, batch_X.T))
    res = np.repeat((C * -batch_y * (f1 / (1.0 + f1))).reshape(batch_X.shape[0], 1), batch_X.shape[1], axis=1) * batch_X
    # print 'res.shape: ', res.shape
    # print 'res.sum(axis=0) shape: ', res.sum(axis=0).shape
    return grad + res.sum(axis=0)

def smoothing_optimizator(X, y, lambd, C, max_iter, eps, alpha, decay, batch_size):
    k = 0
    w = np.zeros(X.shape[1])
    f1 = open('outputfile', 'w+')
    
    batch_iter = 0
    while True:
        w_update = alpha * smoothing_grad_descent(X, y, w, lambd, C, batch_size, batch_iter)
        w -= w_update
        alpha -= alpha * decay
        k += 1
        # if k % 200 == 0:
        #    print "smoothing_optimizator k: ", k
        batch_iter = batch_iter + 1
        if k >= max_iter or np.linalg.norm(w_update, ord=2) < eps:
            break
    return k, w

class Smoothing_Regularization(BaseEstimator, LinearClassifierMixin):

    def __init__(self, C=1.,lambd=1., max_iter=1000, eps=0.0001, alpha=0.01, decay=0.01, fit_intercept=True, batch_size=100):
        self.C = C
        self.lambd = lambd
        self.max_iter = max_iter
        self.eps = eps
        self.alpha = alpha
        self.decay = decay
        self.fit_intercept = fit_intercept
        self.batch_size = batch_size

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        y = 2. * y - 1
        if self.fit_intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        self.n_iter_, self.w_, = smoothing_optimizator(X, y, self.lambd, self.C,
                                                     self.max_iter, self.eps, self.alpha, self.decay, self.batch_size)
        self.coef_ = self.w_.reshape((1, X.shape[1]))
        self.intercept_ = 0.0
        if self.fit_intercept:
            self.intercept_ = self.coef_[:,-1]
            self.coef_ = self.coef_[:,:-1]
        return self
    
    def predict_proba(self, X):
        return super(Smoothing_Regularization, self)._predict_proba_lr(X)
    
    def get_params(self, deep=True):
        return {'lambd': self.lambd,
                'C': self.C,
                'max_iter': self.max_iter, 
                'eps': self.eps, 
                'alpha': self.alpha,
                'decay': self.decay,
                'fit_intercept': self.fit_intercept}
