# (0) overall logic
# (1) better coding method?
# (2) correct? compared to dense
# (3) all are csr matrix?
# sparse: w + v
# sparse: linalg.norm(no ord)
# ?? batchsize * batch_iter / shape[0]?
# gradient descent: divided by # of samples --> float?
__author__ = "Luo Zhaojing"
__version__ = "0.1.0"

import numpy as np
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
import random
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
def huber_grad_descent_avg(batch_X, batch_y, w, v, param, C, is_l1):
    # sparse: if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in huber gd avg"
    # print "batch_X norm: ", np.linalg.norm(batch_X)
    # print "w shape: ", w.shape
    # print "batch_X shape: ", batch_X.shape
    batch_y = batch_y.T
    # print "batch_y shape: ", batch_y.shape
    # print "(-batch_y * (w + v).dot(batch_X.T) shape: ", (-batch_y * (w + v).dot(batch_X.T)).shape
    grad = param * v.sign() if is_l1 else param * 2.0 * w
    # print "-batch_y shape: ",  (-batch_y).shape
    # print "(w + v) shape: ", (w + v).shape
    # print "((w + v).dot(batch_X.T)) shape: ", ((w + v).dot(batch_X.T)).shape
    f1 = np.exp(((-batch_y).multiply((w + v).dot(batch_X.T))).toarray())
    # print "f1 shape: ", f1.shape
    # print "batch_X.T shape: ", batch_X.T.shape
    # print "batch_y shape", batch_y.shape
    # print "(f1 / (1.0 + f1)) shape: ", (f1 / (1.0 + f1)).shape
    # print "(C * (-1 * batch_y.T).toarray() shape: ", (C * -batch_y).toarray().shape
    # print "(C * ((-batch_y).multiply(f1 / (1.0 + f1)))) shape: ", (C * ((-batch_y).multiply(f1 / (1.0 + f1)))).shape
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    # print "y_res shape: ", y_res.shape
    # print "f1 shape: ", f1.shape
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    # y_res = y_res.flatten()
    # print "y_res shape: ", y_res.shape
    # print "y_res: ", y_res
    # print "isspmatrix_csr(y_res): ", sparse.isspmatrix_csr(y_res)
    # print "issparse(y_res): ", sparse.issparse(y_res)
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    # print "res shape: ", res.shape
    res = res.T
    # print "isspmatrix_csr(res): ", sparse.isspmatrix_csr(res)
    #print "issparse(res): ", sparse.issparse(res)
    ressum = res.sum(axis=0) #this will turn back to dense
    # print "isspmatrix_csr(ressum): ", sparse.isspmatrix_csr(ressum)
    # print "issparse(ressum): ", sparse.issparse(ressum)
    ressum = ressum.astype(np.float)
    # print "isspmatrix_csr(ressum): ", sparse.isspmatrix_csr(ressum)
    ressum /=  float(batch_X.shape[0])
    # print "grad shape: ", grad.shape
    # print "ressum shape: ", ressum.shape
    # print "isspmatrix_csr(grad + ressum)", sparse.isspmatrix_csr(grad + ressum)
    # print "grad + sparse.csr_matrix(ressum) norm: ", linalg.norm(grad + sparse.csr_matrix(ressum))
    return grad + sparse.csr_matrix(ressum)

def lasso_grad_descent_avg(batch_X, batch_y, w, param, l1_ratio_or_mu, C):
    #if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in lasso gd avg"
    batch_y = batch_y.T
    grad = param * w.sign()
    f1 = np.exp(((-batch_y).multiply(w.dot(batch_X.T))).toarray())
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    res = res.T
    ressum = res.sum(axis=0) #this will turn back to dense
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return grad + sparse.csr_matrix(ressum)

def ridge_grad_descent_avg(batch_X, batch_y, w, param, l1_ratio_or_mu, C):
    # if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in ridge gd avg"
    batch_y = batch_y.T
    grad = param * 2.0 * w
    f1 = np.exp(((-batch_y).multiply(w.dot(batch_X.T))).toarray())
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    res = res.T
    ressum = res.sum(axis=0) #this will turn back to dense
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return grad + sparse.csr_matrix(ressum)

def elasticnet_grad_descent_avg(batch_X, batch_y, w, param, l1_ratio_or_mu, C):
    # if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in elastic gd avg"
    batch_y = batch_y.T
    grad = param * l1_ratio_or_mu * w.sign() + param * (1 - l1_ratio_or_mu) * w
    f1 = np.exp(((-batch_y).multiply(w.dot(batch_X.T))).toarray())
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    res = res.T
    ressum = res.sum(axis=0) #this will turn back to dense
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return grad + sparse.csr_matrix(ressum)


def smoothing_grad_descent_avg(batch_X, batch_y, w, param, l1_ratio_or_mu, C):
    # print "begin toarray"
    # if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "end toarray"
    # print "in smoothing gd avg"
    batch_y = batch_y.T
    w_array = w.toarray()
    grad =  param * (np.exp(w_array) - 1) / (np.exp(w_array) + 1) # log(1+e^(-w)) + log(1+e^(w))
    # print "grad: ", grad
    # print "float(batch_X.shape[0]): ", float(batch_X.shape[0])
    # print 'grad.shape: ', grad.shape
    f1 = np.exp(((-batch_y).multiply(w.dot(batch_X.T))).toarray())
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    res = res.T
    ressum = res.sum(axis=0) #this will turn back to dense
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return sparse.csr_matrix(grad + ressum)

def huber_optimizator_avg(X, y, lambd, l1_ratio_or_mu, C, max_iter, eps, alpha, decay, batch_size, clf_name):
    k = 0
    w = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])
    w = sparse.csr_matrix(w)
    v = sparse.csr_matrix(v)
    y = sparse.csr_matrix(y)
    y = y.T
    X = sparse.csr_matrix(X)

    batch_iter = 0
    np.random.seed(10)
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]

    grad_descent_avg = huber_grad_descent_avg
    while True:
        index = (batch_size * batch_iter) % X.shape[0]

        if (index + batch_size) >= X.shape[0]: #new epoch
            index = 0
            batch_iter = 0 #new epoch
            np.random.seed(k)
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]

        batch_X, batch_y = X[index : (index + batch_size)], y[index : (index + batch_size)]

        vec = (w + v)
        # making optimization in w and v
        # w and v will be automatically changed to sparse array
        v -= alpha * grad_descent_avg(batch_X, batch_y, w, v, l1_ratio_or_mu, C, True)
        # print "isspmatrix_csr(v): ", sparse.isspmatrix_csr(v)
        w -= alpha * grad_descent_avg(batch_X, batch_y, w, v, lambd, C, False)
        # v_update = alpha * grad_descent_avg(batch_X, batch_y, w, v, l1_ratio_or_mu, C, True)
        # v -= v_update
        # print "v_update norm: ", np.linalg.norm(v_update)
        # w_update = alpha * grad_descent_avg(batch_X, batch_y, w, v, lambd, C, False)
        # w -= w_update
        # print "w_update norm: ", np.linalg.norm(w_update)

        alpha -= alpha * decay
        k += 1
        if k % 20 == 0:
            print "huber_optimizator k: ", k
        batch_iter = batch_iter + 1
        if k >= max_iter or linalg.norm(((w+v) - vec)) < eps:
            break
    print "huber opt avg final k: ", k
    return k, w.toarray(), v.toarray()

# only one weight w, not w + v
def non_huber_optimizator_avg(X, y, lambd, l1_ratio_or_mu, C, max_iter, eps, alpha, decay, batch_size, clf_name):
    k = 0
    w = np.zeros(X.shape[1])
    w = sparse.csr_matrix(w)
    y = sparse.csr_matrix(y)
    y = y.T
    X = sparse.csr_matrix(X)
    # print "w shape: ", w.shape
    # f1 = open('outputfile', 'w+')

    batch_iter = 0
    np.random.seed(10)
    idx = np.random.permutation(X.shape[0])
    print "data idx: ", idx
    X = X[idx]
    y = y[idx]
    if clf_name == 'lasso':
        grad_descent_avg = lasso_grad_descent_avg
    elif clf_name == 'ridge':
        grad_descent_avg = ridge_grad_descent_avg
    elif clf_name == 'elasticnet':
        grad_descent_avg = elasticnet_grad_descent_avg
    elif clf_name == 'smoothing':
        grad_descent_avg = smoothing_grad_descent_avg
    while True:
        # sparse matrix works, random.shuffle
        # shuffle: next time shuffle index will be forgetten (static variable: smoothing_grad_descent.idx)
        # print "X.shape: ", X.shape
        # print "max idx: ", max(idx)
        index = (batch_size * batch_iter) % X.shape[0]

        if (index + batch_size) >= X.shape[0]: #new epoch
            index = 0
            batch_iter = 0 #new epoch
            np.random.seed(k)
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]

        batch_X, batch_y = X[index : (index + batch_size)], y[index : (index + batch_size)]
        w_update = alpha * grad_descent_avg(batch_X, batch_y, w, lambd, l1_ratio_or_mu, C)
        print "w_update norm: ", linalg.norm(w_update)
        w -= w_update
        alpha -= alpha * decay
        k += 1
        # if k % 20 == 0:
        print "non_huber_optimizator k: ", k
        batch_iter = batch_iter + 1
        if k >= max_iter or linalg.norm(w_update) < eps:
            break
    print "non_huber opt avg final k: ", k
    return k, w.toarray()
