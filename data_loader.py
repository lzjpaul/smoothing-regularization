import numpy as np
# from matplotlib import pyplot as plt
import pickle
from gm_prior_simulation import Simulator
from pickle_transformer import Dataset
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_svmlight_file

# load training/testing data from pickles simulator object file with specified training percent
def loadData(fileName, onehot=True, sparsify=True):
    if 'pkl' in fileName:
        print '\n===============================================\n'
        print 'loading pkl data...'
        pklfile = pickle.load(open(fileName, 'r'))
        X, Y = pklfile.x, pklfile.label.reshape((pklfile.sample_num, 1))
        print 'finish loading data...\ndata samples %d' %(pklfile.sample_num)
        print 'data dimension %d' %(pklfile.dimension)
        print '\n===============================================\n'
        print "X shape: ", X.shape
        print "Y shape: ", Y.shape
        print "Y[:10] before transform: ", Y[:10]
        if Y.dtype != 'bool':
            Y = (Y > 0.5)
        print "Y[:10] after transform: ", Y[:10]
        np.random.seed(10)
        idx = np.random.permutation(X.shape[0])
        print "idx: ", idx
        X = X[idx]
        Y = Y[idx]
        # return X, Y
        if onehot:
            return (OneHotEncoder().fit_transform(X, ), Y) if sparsify==True else (OneHotEncoder().fit_transform(X, ).toarray(), Y)
        else:
            return (sparse.csr_matrix(X), Y) if sparsify==True else (X, Y)
    else:
        print '\n===============================================\n'
        print 'loading partial svm data...'
        data = load_svmlight_file(fileName)
        X, Y = data[0], data[1]
        # print "svmlight X shape: ", X.shape
        # print "svmlight Y shape: ", Y.shape
        Y = (Y > 0.5)
        Y = Y.reshape((-1, 1))
        print "Y[:10] after transform: ", Y[:10]
        np.random.seed(10)
        idx = np.random.permutation(X.shape[0])
        print "idx: ", idx
        X = X[idx]
        Y = Y[idx]
        # return partial X, Y
        partial_perc = 0.5
        partialNum = int(partial_perc*X.shape[0])
        print "partialNum: ", partialNum
        return X[:partialNum, ], Y[:partialNum, ]
