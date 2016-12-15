__author__ = "Gunthard Benecke and Oleksandr Zadorozhnyi"
__license__ = "GPL"
__version__ = "0.2.0"

import numpy as np
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator

def grad_descent(X, y, w, v, param, C, is_l1):
    grad = param * np.sign(v) if is_l1 else param * 2.0 * w
    f1 = np.exp(-y * np.dot(w + v, X.T))
    res = np.repeat((C * -y * (f1 / (1.0 + f1))).reshape(X.shape[0], 1), X.shape[1], axis=1) * X
    return grad + res.sum(axis=0)

def optimizator(X, y, lambd, mu, C, max_iter, eps, alpha, decay):
    k = 0
    w = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])
    
    while True:
        vec = np.add(w, v)
        
        # making optimization in w and v
        v -= alpha * grad_descent(X, y, w, v, mu, C, True)
        w -= alpha * grad_descent(X, y, w, v, lambd, C, False)
        
        alpha -= alpha * decay
        k += 1

        if k >= max_iter or np.linalg.norm(np.add(w,v) - vec, ord=2) < eps:
            break

    return k, w, v

class HuberSVC(BaseEstimator, LinearClassifierMixin):

    def __init__(self, lambd=1., mu=1., C=1., max_iter=1000, eps=0.0001, alpha=0.01, decay=0.01, fit_intercept=True):
        self.lambd = lambd
        self.mu = mu
        self.C = C
        self.max_iter = max_iter
        self.eps = eps
        self.alpha = alpha
        self.decay = decay
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        y = 2. * y - 1
        #if self.fit_intercept:
        #    X = np.hstack((X, np.ones((X.shape[0], 1))))
        if self.fit_intercept:
            if sparse.issparse(X):
                # X = np.hstack((X.toarray(), np.ones((X.shape[0], 1))))
                # X = csr_matrix(X)
                # X = sparse.hstack((X, csr_matrix(np.ones((X.shape[0], 1))))).tocsr()
                X = sparse.hstack([X, np.ones((X.shape[0], 1))], format="csr")
            else:
                X = np.hstack((X, np.ones((X.shape[0], 1))))
        self.n_iter_, self.w_, self.v_ = optimizator(X, y, self.lambd, self.mu, self.C, 
                                                     self.max_iter, self.eps, self.alpha, self.decay)
        self.coef_ = np.add(self.w_, self.v_).reshape((1, X.shape[1]))
        self.intercept_ = 0.0
        if self.fit_intercept:
            self.intercept_ = self.coef_[:,-1]
            self.coef_ = self.coef_[:,:-1]
        return self
    
    def predict_proba(self, X):
        return super(HuberSVC, self)._predict_proba_lr(X)
    
    def get_params(self, deep=True):
        return {'lambd': self.lambd, 
                'mu': self.mu, 
                'C': self.C, 
                'max_iter': self.max_iter, 
                'eps': self.eps, 
                'alpha': self.alpha,
                'decay': self.decay,
                'fit_intercept': self.fit_intercept}
