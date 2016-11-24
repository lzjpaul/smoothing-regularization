#11-23 adding parama lambda for tuning parameters
__author__ = "Luo Zhaojing"
__version__ = "0.1.0"

import numpy as np
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator

def smoothing_grad_descent(X, y, w, param, C, batch_size):
    # debug
    #print "SGD\n"
    #randomIndex = np.random.random_integers(0, X.shape[0]-1, batch_size)
    #X, y = X[randomIndex], y[randomIndex]       # random sample #batch_size samples - SGD
    grad =  param * (np.exp(w) - 1) / (np.exp(w) + 1) # log(1+e^(-w)) + log(1+e^(w))
    # print 'grad.shape: ', grad.shape
    f1 = np.exp(-y * np.dot(w, X.T))
    res = np.repeat((C * -y * (f1 / (1.0 + f1))).reshape(X.shape[0], 1), X.shape[1], axis=1) * X
    # print 'res.shape: ', res.shape
    # print 'res.sum(axis=0) shape: ', res.sum(axis=0).shape
    return grad + res.sum(axis=0)

def smoothing_optimizator(X, y, lambd, C, max_iter, eps, alpha, decay, batch_size):
    k = 0
    w = np.zeros(X.shape[1])
    f1 = open('outputfile', 'w+')

    while True:
        vec = np.copy(w)
        
        # making optimization in w
        # print 'vec: ', vec
        w -= alpha * smoothing_grad_descent(X, y, w, lambd, C, batch_size)
        # print 'after w: ', w

        alpha -= alpha * decay
        k += 1

        print >> f1, "w: ", w
        print >> f1, "vec", vec
        print >> f1, "np.linalg.norm(w - vec, ord=2): ", np.linalg.norm(w - vec, ord=2)
        print >> f1, "k: ", k
        if k >= max_iter or np.linalg.norm(w - vec, ord=2) < eps:
            break
        print 'k = ', k
    return k, w

class Smoothing_Regularization(BaseEstimator, LinearClassifierMixin):

    def __init__(self, C=1.,lambd=1., max_iter=1000, eps=0.0001, alpha=0.01, decay=0.01, fit_intercept=True, batch_size=20):
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
