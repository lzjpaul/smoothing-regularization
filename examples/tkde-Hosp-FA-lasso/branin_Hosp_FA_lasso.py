import numpy as np
import math
import sys
# from data_loader import *
# from pickle_transformer import Dataset
import argparse
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import sparse
from scipy.sparse import linalg
import datetime
import time
from sklearn.utils.extmath import safe_sparse_dot
import pandas
import pickle

class Dataset(object):
    def DataGenerator(self, x, y):
        self.sample_num, self.dimension = x.shape[0], x.shape[1]
        self.x = x
        self.label = y

class Simulator():
    def __init__(self, gm_num, dimension, sample_num, pi, variance, covariace):
        # init simulator parameters
        self.gm_num, self.dimension, self.sample_num = gm_num, dimension, sample_num
        self.pi, self.variance, self.covariance = pi, variance, covariace

    def wGenerator(self):
        # generate w_origin from mixture coefficient
        self.w_origin = np.random.choice(4, size=(self.dimension), p=self.pi)
        self.w = np.ndarray(shape=(self.dimension), dtype='float32')

        # generate w with corresponding gaussian variance
        gm_count = np.bincount(self.w_origin)
        for gm_index in xrange(self.gm_num):
            self.w[self.w_origin==gm_index] = np.random.normal(0.0, np.sqrt(self.variance[gm_index]), size=(gm_count[gm_index]))

    def xGenerator(self):
        self.x = np.random.multivariate_normal(mean=np.zeros(shape=(self.dimension)), cov=self.covariance, size=(self.sample_num,))

    def labelGenerator(self, noiseVar=0.1):
        lg, noise = np.dot(self.x, self.w), np.random.normal(0.0, np.sqrt(noiseVar), size=(self.sample_num))
        y_vals_no_noise =  1/(1+np.exp(-lg))
        y_vals = 1/(1+np.exp(-(lg+noise)))    # adding the gaussian noise term

        uniform_vals = np.random.uniform(low=0.0, high=1.0, size=(self.sample_num))
        self.label_no_noise = (y_vals_no_noise >= uniform_vals)
        self.label = (y_vals>=uniform_vals)
        print "optimal accuracy: ", np.sum((lg >= 0) != self.label) / float(self.sample_num)

        # save generated y_vals_noise
        with open('generated_y_vals.dta', 'w') as saveFile:
            for index, item in enumerate(self.label):
                saveFile.write("%d: %.6f > %.6f\t(yvals %.6f,\tyvals_noise %.6f,\tnoise %10.6f)\n"
                               %(item, y_vals[index], uniform_vals[index], y_vals_no_noise[index], y_vals[index], noise[index]))

        # get probability distribution and noise effect
        print "probability distribution:\t", (np.bincount((y_vals_no_noise*10).astype(int)).astype(float)/self.sample_num)
        print "noise misclassification rate:\t%.6f" %(float(np.sum(self.label!=self.label_no_noise))/self.sample_num)


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
        # print "Y[:10] before transform: ", Y[:10]
        if Y.dtype != 'bool':
            Y = (Y > 0.5)
        # print "Y[:10] after transform: ", Y[:10]
        np.random.seed(10)
        idx = np.random.permutation(X.shape[0])
       #  print "idx: ", idx
        X = X[idx]
        Y = Y[idx]
        # return X, Y
        if onehot:
            return (OneHotEncoder().fit_transform(X.astype(int), ), Y) if sparsify==True else (OneHotEncoder().fit_transform(X.astype(int), ).toarray(), Y)
        else:
            return (sparse.csr_matrix(X), Y) if sparsify==True else (X, Y)
    else:
        print '\n===============================================\n'
        print 'loading full svm data...'
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
        partial_perc = 1.0
        partialNum = int(partial_perc*X.shape[0])
        print "partialNum: ", partialNum
        return X[:partialNum, ], Y[:partialNum, ]

# base logistic regression class
class Logistic_Regression(object):
    def __init__(self, reg_lambda=1, learning_rate=0.1, max_iter=1000, eps=1e-4, batch_size=-1, validation_perc=0.0):
        self.reg_lambda, self.learning_rate, self.max_iter = reg_lambda, learning_rate, max_iter
        self.eps, self.batch_size, self.validation_perc = eps, batch_size, validation_perc
        print "self.reg_lambda, self.learning_rate, self.max_iter: ", self.reg_lambda, self.learning_rate, self.max_iter
        print "self.eps, self.batch_size, self.validation_perc: ", self.eps, self.batch_size, self.validation_perc

    def w_lr(self, epoch):
        #lr_decay = 0.01
        #lr_decay_epoch = 20
        #return self.learning_rate * np.power((1-lr_decay),epoch/float(lr_decay_epoch))
        return self.learning_rate

    def likelihood_grad(self, xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method):
        xTrain, yTrain = xTrain[index : (index + self.batch_size)], yTrain[index : (index + self.batch_size)]

        mu = self.sigmoid(safe_sparse_dot(xTrain, self.w, dense_output=True))
        # check here, no regularization over bias term # need normalization with xTrain.shape[0]/batch_size here
        grad_w = (self.trainNum/self.batch_size)*(safe_sparse_dot(xTrain.T, (mu - yTrain), dense_output=True))

        return grad_w
    # calc the delta w to update w, using sgd here
    def delta_w(self, xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method):
        grad_w = self.likelihood_grad(xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method)
        reg_grad_w = self.reg_lambda * self.w
        # if iter_num < 100 or iter_num % 100 ==0:
        if iter_num % 100000 ==0:
            print "grad_w norm: ", np.linalg.norm(grad_w)
            print "reg_grad_w norm: ", np.linalg.norm(reg_grad_w)
        reg_grad_w[-1, 0] = 0.0 # bias
        grad_w += reg_grad_w
        return -grad_w


    def fit(self, xTrain, yTrain, xTest, yTest, sparsify, ishuber=False, gm_opt_method=-1, verbos=False):
        # find the number of class and feature, allocate memory for model parameters
        self.trainNum, self.featureNum = xTrain.shape[0], xTrain.shape[1]
        if ishuber:
            np.random.seed(10)
            self.w1 = np.random.normal(0, 0.01, size=(self.featureNum+1, 1))#np.zeros(shape=(self.featureNum+1, 1), dtype='float32')
            self.w2 = np.random.normal(0, 0.01, size=(self.featureNum+1, 1))#np.zeros(shape=(self.featureNum+1, 1), dtype='float32')
            self.w = np.add(self.w1, self.w2)
        else:
            np.random.seed(10)
            self.w = np.random.normal(0, 0.01, size=(self.featureNum+1, 1))#np.zeros(shape=(self.featureNum+1, 1), dtype='float32')
        # print "self.w[:10]: ", self.w[:10]

        # adding 1s to each training examples
        if sparsify:
            xTrain = sparse.hstack([xTrain, np.ones(shape=(self.trainNum, 1))], format="csr")
        else:
            xTrain = np.hstack((xTrain, np.ones(shape=(self.trainNum, 1))))

        try:
            iter, pre_train_loss = 0, 0.0
            # Contains the last time each feature was used
            if sparsify:
                self.u = (np.zeros(self.featureNum) - np.ones(self.featureNum)).astype(int)
                print "self.u: ", self.u
                if hasattr(self, 'pi'):
                    self.gm_prior_u = -1
            # minibatch initialization
            batch_iter = 0
            while True:
                # minibatch calculation
                index = self.batch_size * batch_iter
                if (index + self.batch_size) > xTrain.shape[0]:  # new epoch
                    index = 0
                    batch_iter = 0  # new epoch
                    np.random.seed(iter)
                    idx = np.random.permutation(xTrain.shape[0])
                    xTrain = xTrain[idx]
                    yTrain = yTrain[idx]

                # calc current epoch
                epoch_num = iter*self.batch_size/xTrain.shape[0]
                if ishuber:
                    # calc the delta_w1 to update w1
                    delta_w1 = self.delta_w1(xTrain, yTrain, index, epoch_num, iter, gm_opt_method)
                    # update w1
                    self.w1 += self.w_lr(epoch_num) * delta_w1
                    self.w = np.add(self.w1, self.w2)
                    # calc the delta_w2 to update w2
                    delta_w2 = self.delta_w2(xTrain, yTrain, index, epoch_num, iter, gm_opt_method)
                    # update w2
                    self.w2 += self.w_lr(epoch_num) * delta_w2
                    self.w = np.add(self.w1, self.w2)
                else:
                    # calc the delta_w to update w
                    delta_w = self.delta_w(xTrain, yTrain, index, epoch_num, iter, gm_opt_method)
                    # update w
                    self.w += self.w_lr(epoch_num) * delta_w

                # stop updating if converge or nan encountered
                # https://www.coursera.org/learn/machine-learning/lecture/fKi0M/stochastic-gradient-descent-convergence
                iter += 1
                batch_iter += 1
                if iter % 100000 == 0:
                    # print test metrics -- not allowed
                    print "\n\ntest accuracy: %.6f\t|\ttest auc: %6f\t|\ttest loss: %6f" % (self.accuracy(self.predict(xTest, sparsify), yTest), \
                                                               self.auroc(self.predict_proba(xTest, sparsify), yTest), self.loss(xTest, yTest, sparsify))
                    # print test metrics -- not allowed
                    train_loss = self.loss(xTrain, yTrain, sparsify)
                    # print "w norm %10.6f\t|\tdelta_w norm %10.6f\t"%(np.linalg.norm(self.w1), np.linalg.norm(self.w_lr(epoch_num) * delta_w1))
                    print "train_loss %10.10f abs(train_loss - pre_train_loss) %10.10f self.eps %10.10f"%(train_loss, abs(train_loss - pre_train_loss), self.eps)
                    if not ishuber:
                        print "w norm %10.6f\t|\tdelta_w norm %10.6f"%(np.linalg.norm(self.w), np.linalg.norm(self.w_lr(epoch_num) * delta_w))
                    else:
                        print "w1 norm %10.6f\t|\tdelta_w1 norm %10.6f"%(np.linalg.norm(self.w1), np.linalg.norm(self.w_lr(epoch_num) * delta_w1))
                        print "w2 norm %10.6f\t|\tdelta_w2 norm %10.6f"%(np.linalg.norm(self.w2), np.linalg.norm(self.w_lr(epoch_num) * delta_w2))
                        print "w norm %10.6f"%(np.linalg.norm(self.w))
                    if iter > self.max_iter or abs(train_loss - pre_train_loss) < self.eps:
                        break
                    pre_train_loss = train_loss
                if np.isnan(np.linalg.norm(self.w)):
                    print "iter %4d\tw norm is nan"%(iter)
                    break
                # print useful information
                if iter % 100000 == 0:
                    # print np.sum(np.abs(self.w))/self.featureNum, np.linalg.norm(self.w, ord=2)
                    train_accuracy = self.accuracy(self.predict(xTrain, sparsify), yTrain)
                    train_loss = self.loss(xTrain, yTrain, sparsify)
                    if verbos:
                        print "iter %4d\t|\ttrain_accuracy %10.6f\t|\ttrain_loss %10.10f"%(iter, train_accuracy, train_loss)
                        if hasattr(self, 'pi'):
                            regularization_loss = self.w_loss()
                            print "w norm %10.6f\t|\tdelta_w norm %10.6f\t|\tw_loss %10.10f"%(np.linalg.norm(self.w), np.linalg.norm(self.w_lr(epoch_num) * delta_w), regularization_loss)
                            print "lr %8.6f\t|\toverall loss %10.10f"%(self.w_lr(epoch_num), (train_loss+regularization_loss))
                            print "pi, reg_lambda: ", self.pi, self.reg_lambda
                            print "lr, pi_r_l, reg_lambda_s_lr: ",self.w_lr(epoch_num), self.pi_r_lr(epoch_num), self.reg_lambda_s_lr(epoch_num)
                            print "\n"
        finally:
            self.w = self.w

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
        return np.sum(yPredict == yTrue) / float(yTrue.shape[0])

    def auroc(self, yPredictProba, yTrue):
        return roc_auc_score(yTrue, yPredictProba)

    # sigmoid function
    def sigmoid(self, matrix):
        return 1.0/(1.0+np.exp(-matrix))

    # model parameter
    def __str__(self):
        return 'model config {\treg: %.6f, lr: %.6f, batch_size: %5d\t}' \
            % (self.reg_lambda, self.learning_rate, self.batch_size)

# base logistic regression class
class Lasso_Logistic_Regression(Logistic_Regression):
    def __init__(self, reg_lambda=1, learning_rate=0.1, max_iter=1000, eps=1e-4, batch_size=-1, validation_perc=0.0):
        Logistic_Regression.__init__(self, reg_lambda, learning_rate, max_iter, eps, batch_size, validation_perc)

    # calc the delta w to update w, using sgd here
    def delta_w(self, xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method):
        grad_w = self.likelihood_grad(xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method)
        reg_grad_w = self.reg_lambda * np.sign(self.w)
        if iter_num % 100000 ==0:
            print "grad_w norm: ", np.linalg.norm(grad_w)
            print "reg_grad_w norm: ", np.linalg.norm(reg_grad_w)
        reg_grad_w[-1, 0] = 0.0 # bias
        grad_w += reg_grad_w
        return -grad_w

    # model parameter
    def __str__(self):
        return 'model config {\treg: %.6f, lr: %.6f, batch_size: %5d\t}' \
            % (self.reg_lambda, self.learning_rate, self.batch_size)


def branin(reg_lambda_value, datapath, onehot, sparsify, batchsize, wlr, maxiter):
    print ('brain.py brain() reg_lambda_value: ', reg_lambda_value)
    sys.stderr.write("simple/brain.py brain()\n")
    sys.stderr.write ("simple/brain.py brain() time: %s \n" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    # begins from here, loading the permutated data
    x, y = loadData(datapath, onehot=(onehot==1), sparsify=(sparsify==1))
    print "loadData x shape: ", x.shape
    n_folds = 5
    accuracy_df = pandas.DataFrame()
    auc_df = pandas.DataFrame()
    loss_df = pandas.DataFrame()
    for i, (train_index, test_index) in enumerate(StratifiedKFold(y.reshape(y.shape[0]), n_folds=n_folds)):
        # if i > 0:
        #     break
        print "subsample i: ", i
        reg_lambda = [reg_lambda_value]
        for reg in reg_lambda:
            start = time.time()
            st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
            print st
            # print "train_index: ", train_index
            # print "test_index: ", test_index
            xTrain, yTrain, xTest, yTest = x[train_index], y[train_index], x[test_index], y[test_index]
            learning_rate, max_iter = math.pow(10, (-1 * wlr)), maxiter
            eps, batch_size = 1e-10, batchsize
            print "\nreg_lambda: %f" % (reg)
            LG = Lasso_Logistic_Regression(reg, learning_rate, max_iter, eps, batch_size)
            LG.fit(xTrain, yTrain, xTest, yTest, (sparsify==1), gm_opt_method=-1, verbos=True)
            if not np.isnan(np.linalg.norm(LG.w)):
                print "\n\nfinal accuracy: %.6f\t|\tfinal auc: %6f\t|\ttest loss: %6f" % (LG.accuracy(LG.predict(xTest, (sparsify==1)), yTest), \
                                                               LG.auroc(LG.predict_proba(xTest, (sparsify==1)), yTest), LG.loss(xTest, yTest, (sparsify==1)))
                accuracy_df.loc[i, (str(reg))] = LG.accuracy(LG.predict(xTest, (sparsify==1)), yTest)
                auc_df.loc[i, (str(reg))] = LG.auroc(LG.predict_proba(xTest, (sparsify==1)), yTest)
                loss_df.loc[i, (str(reg))] = (LG.loss(xTest, yTest, (sparsify==1)) * xTest.shape[0])
            print LG

            # plt.hist(LG.w, bins=50, normed=1, color='g', alpha=0.75)
            # plt.show()
            done = time.time()
            do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
            print do
            elapsed = done - start
            print elapsed
            # np.savetxt('weight-out/'+sys.argv[0][:-3]+'_w.out', LG.w, delimiter=',')

    print accuracy_df
    # pandas.options.display.float_format = '{:,.6f}'.format
    # print accuracy_df.values
    # print "\n\naccuracy pandas results\n\n"
    # print "mean: ", accuracy_df.mean().values
    # print "std: ", accuracy_df.std().values
    # print "max mean index: ", accuracy_df.mean().idxmax()
    # print "max mean: ", accuracy_df.mean().max()
    # print "max mean std: ", accuracy_df.std().loc[accuracy_df.mean().idxmax()]
    print("accuracy best parameter %0.6f (+/-%0.06f)"
                          % (accuracy_df.mean().max(), accuracy_df.std().loc[accuracy_df.mean().idxmax()]))
    # print "max mean std: ", result_df.loc['Std'].loc[result_df['Mean'].idxmax()]
    # print "best each subsample: \n", accuracy_df.max(axis=1)
    # print "best each subsample index: \n", accuracy_df.idxmax(axis=1)
    # print "mean of each subsample best: ", accuracy_df.max(axis=1).mean()
    # print "std of each subsample best: ", accuracy_df.max(axis=1).std()
    # print("accuracy best subsample %0.6f (+/-%0.06f)"
    #                       % (accuracy_df.max(axis=1).mean(), accuracy_df.max(axis=1).std()))

    print loss_df
    # pandas.options.display.float_format = '{:,.6f}'.format
    # print loss_df.values
    # print "\n\nloss pandas results\n\n"
    # print "mean: ", loss_df.mean().values
    # print "std: ", loss_df.std().values
    # print "min mean index: ", loss_df.mean().idxmin()
    # print "min mean: ", loss_df.mean().min()
    # print "min mean std: ", loss_df.std().loc[loss_df.mean().idxmin()]
    print("loss best parameter %0.6f (+/-%0.06f)"
                         % (loss_df.mean().min(), loss_df.std().loc[loss_df.mean().idxmin()]))
    # print "min mean std: ", result_df.loc['Std'].loc[result_df['Mean'].idxmin()]
    # print "best each subsample: \n", loss_df.min(axis=1)
    # print "best each subsample index: \n", loss_df.idxmin(axis=1)
    # print "mean of each subsample best: ", loss_df.min(axis=1).mean()
    # print "std of each subsample best: ", loss_df.min(axis=1).std()
    # print("loss best subsample %0.6f (+/-%0.06f)"
    #                      % (loss_df.min(axis=1).mean(), loss_df.min(axis=1).std()))

    # loss for this parameter parameter
    result = float(loss_df.mean().min())
    # result = (1-accuracy_df.mean().max())
    # sys.stderr.write ('Result = %f\n' % result)
    print ('Test Loss Result = %f\n' % result)
    # print ('Error Rate Result = %f\n' % result)
    #time.sleep(np.random.randint(60))
    return result

# Write a function like this called 'main'
def main(job_id, params):
    sys.stderr.write("in brain.py main()\n")
    sys.stderr.write('Anything printed here will end up in the output directory for job #%d\n' % job_id)
    sys.stderr.write('params\n')
    print ("in brain.py main() params: ", params)
    datapath = "/home/singa/zhaojing/smooth-regularization-dbpcm-bak-logistic/spearmint/Spearmint/examples/data/LACE-CNN-1500-lastcase.pkl"
    onehot = 0
    sparsify = 0
    batchsize = 50
    wlr = 4
    maxiter = 120000
    # only one weight decay value
    return branin(params['x'][0], datapath, onehot, sparsify, batchsize, wlr, maxiter)
