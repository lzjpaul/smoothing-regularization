'''
Luo Zhaojing - 2017.3
Logistic Regression
'''
'''
hyper:
(1) lr decay
(2) threshold for train_loss
'''
import sys
from data_loader import *
import argparse
import math
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import sparse
from scipy.sparse import linalg
import datetime
import time
from sklearn.utils.extmath import safe_sparse_dot
import pandas
# base logistic regression class
class Logistic_Regression(object):
    def __init__(self, reg_lambda=1, learning_rate=0.1, max_iter=1000, eps=1e-4, batch_size=-1, validation_perc=0.0):
        self.reg_lambda, self.learning_rate, self.max_iter = reg_lambda, learning_rate, max_iter
        self.eps, self.batch_size, self.validation_perc = eps, batch_size, validation_perc
        print "self.reg_lambda, self.learning_rate, self.max_iter: ", self.reg_lambda, self.learning_rate, self.max_iter
        print "self.eps, self.batch_size, self.validation_perc: ", self.eps, self.batch_size, self.validation_perc
    # loss function
    def loss(self, samples, yTrue, sparsify):
        if samples.shape[1] != self.w.shape[0]:
            if sparsify:
                samples = sparse.hstack([samples, np.ones(shape=(samples.shape[0], 1))], format="csr")
            else:
                samples = np.hstack((samples, np.ones(shape=(samples.shape[0], 1))))
        threshold = 1e-320
        yTrue = yTrue.astype(int)
        mu = self.sigmoid(safe_sparse_dot(samples, self.w, dense_output=True))
        mu_false = (1-mu)
        return np.sum((-yTrue * np.log(np.piecewise(mu, [mu < threshold, mu >= threshold], [threshold, lambda mu:mu])) \
                       - (1-yTrue) * np.log(np.piecewise(mu_false, [mu_false < threshold, mu_false >= threshold], [threshold, lambda mu_false:mu_false]))), axis = 0) / float(samples.shape[0])

    # predict result
    def predict(self, samples, sparsify):
        if samples.shape[1] != self.w.shape[0]:
            if sparsify:
                samples = sparse.hstack([samples, np.ones(shape=(samples.shape[0], 1))], format="csr")
            else:
                samples = np.hstack((samples, np.ones(shape=(samples.shape[0], 1))))
        return safe_sparse_dot(samples, self.w, dense_output=True)>0.0

    # predict probability
    def predict_proba(self, samples, sparsify):
        if samples.shape[1] != self.w.shape[0]:
            if sparsify:
                samples = sparse.hstack([samples, np.ones(shape=(samples.shape[0], 1))], format="csr")
            else:
                samples = np.hstack((samples, np.ones(shape=(samples.shape[0], 1))))
        return self.sigmoid(safe_sparse_dot(samples, self.w, dense_output=True))

    # calc accuracy
    def accuracy(self, yPredict, yTrue):
        print "yPredict == yTrue number: ", np.sum(yPredict == yTrue)
        return np.sum(yPredict == yTrue) / float(yTrue.shape[0])

    # calc accuracy
    def BER_accuracy(self, yPredict, yTrue):
        print "yTrue: ", yTrue.transpose()
        print "yPredict: ", yPredict.transpose()

        print "True number: ", float(yTrue[yTrue==True].shape[0])

        print "Predict True number: ", float(yPredict[yPredict==True].shape[0])

        print "False number: ", float(yTrue[yTrue==False].shape[0])

        print "Predict False number: ", float(yPredict[yPredict==False].shape[0])

        print "True: ", (np.sum(yPredict[yTrue==True] != yTrue[yTrue==True]) / float(yTrue[yTrue==True].shape[0]))
        print "False: ", (np.sum(yPredict[yTrue==False] != yTrue[yTrue==False]) / float(yTrue[yTrue==False].shape[0]))
        return ( (np.sum(yPredict[yTrue==True] != yTrue[yTrue==True]) / float(yTrue[yTrue==True].shape[0]))\
                + (np.sum(yPredict[yTrue==False] != yTrue[yTrue==False]) / float(yTrue[yTrue==False].shape[0])) ) / 2.0

    def auroc(self, yPredictProba, yTrue):
        return roc_auc_score(yTrue, yPredictProba)

    # sigmoid function
    def sigmoid(self, matrix):
        return 1.0/(1.0+np.exp(-matrix))

    # model parameter
    def __str__(self):
        return 'model config {\treg: %.6f, lr: %.6f, batch_size: %5d\t}' \
            % (self.reg_lambda, self.learning_rate, self.batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-datapath', type=str, help='the dataset path, not svm')
    parser.add_argument('-onehot', type=int, help='need onehot or not')
    parser.add_argument('-sparsify', type=int, help='need sparsify or not')
    parser.add_argument('-weightpath', type=str, help='the weight path, not svm')
    args = parser.parse_args()

    # load the permutated data
    x, y = loadData(args.datapath, onehot=(args.onehot==1), sparsify=(args.sparsify==1))
    print "loadData x shape: ", x.shape
    n_folds = 5
    weight = np.genfromtxt(args.weightpath, delimiter=',').reshape((-1, 1))
    for i, (train_index, test_index) in enumerate(StratifiedKFold(y.reshape(y.shape[0]), n_folds=n_folds)):
        if i > 0:
            break
        print "subsample i: ", i
        xTrain, yTrain, xTest, yTest = x[train_index], y[train_index], x[test_index], y[test_index]
        LG = Logistic_Regression()
        LG.w = weight
        if not np.isnan(np.linalg.norm(LG.w)):
            print "\n\nfinal accuracy: %.6f\t|\tBER: %6f\t|\tfinal auc: %6f\t|\ttest loss: %6f" % (LG.accuracy(LG.predict(xTest, (args.sparsify==1)), yTest), \
            LG.BER_accuracy(LG.predict(xTest, (args.sparsify==1)), yTest), LG.auroc(LG.predict_proba(xTest, (args.sparsify==1)), yTest), LG.loss(xTest, yTest, (args.sparsify==1)))
