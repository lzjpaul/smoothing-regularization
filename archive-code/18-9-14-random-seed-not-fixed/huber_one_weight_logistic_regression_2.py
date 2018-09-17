'''
Luo Zhaojing - 2017.3
Huber One Weight Logistic Regression
'''
'''
hyper:
(1) lr decay
(2) threshold for train_loss
'''
import sys
from logistic_regression import Logistic_Regression
from data_loader import *
import argparse
import math
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import datetime
import time
import pandas
# base logistic regression class
class Huber_One_Weight_Logistic_Regression(Logistic_Regression):
    def __init__(self, reg_mu=1, reg_lambda=1, learning_rate=0.1, max_iter=1000, eps=1e-4, batch_size=-1, validation_perc=0.0):
        Logistic_Regression.__init__(self, reg_lambda, learning_rate, max_iter, eps, batch_size, validation_perc)
        self.reg_mu = reg_mu
        print "self.reg_mu: ", self.reg_mu

    # calc the delta w to update w, using sgd here
    def delta_w(self, xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method):
        grad_w = self.likelihood_grad(xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method)
        w_array = np.copy(self.w)
        threshold = (self.reg_mu)/(self.reg_lambda*2.0)
        reg_grad_w = np.piecewise(w_array, [np.absolute(w_array) < threshold, np.absolute(w_array) >= threshold], \
                                  [lambda w_array: 2*self.reg_lambda*w_array, lambda w_array: self.reg_mu*np.sign(w_array)]).reshape((-1, 1))
        reg_grad_w[-1, 0] = 0.0 # bias
        grad_w += reg_grad_w
        return -grad_w

    # model parameter
    def __str__(self):
        return 'model config {\treg_mu: %.6f, reg_lambda: %.6f, lr: %.6f, batch_size: %5d\t}' \
            % (self.reg_mu, self.reg_lambda, self.learning_rate, self.batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-datapath', type=str, help='the dataset path, not svm')
    parser.add_argument('-onehot', type=int, help='need onehot or not')
    parser.add_argument('-sparsify', type=int, help='need sparsify or not')
    parser.add_argument('-batchsize', type=int, help='batchsize')
    parser.add_argument('-wlr', type=int, help='weight learning_rate (to the power of 10)')
    parser.add_argument('-maxiter', type=int, help='max_iter')
    args = parser.parse_args()

    # load the permutated data
    x, y, yvals = loadData(args.datapath, onehot=(args.onehot==1), sparsify=(args.sparsify==1))
    n_folds = 5
    for i, (train_index, test_index) in enumerate(StratifiedKFold(y.reshape(y.shape[0]), n_folds=n_folds)):
        if i > 0:
            break
        reg_mu = [1e-2, 1e-1]
        reg_lambda = [1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1000.]
        for mu_val in reg_mu:
            for lambda_val in reg_lambda:
                start = time.time()
                st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
                print st
                train_index = range(0, 10000)
                test_index = range(10000, 50000)
                #print "train_index: ", train_index
                #print "test_index: ", test_index
                xTrain, yTrain, xTest, yTest, yvalsTest = x[train_index], y[train_index], x[test_index], y[test_index], yvals[test_index]
                learning_rate, max_iter = math.pow(10, (-1 * args.wlr)), args.maxiter
                eps, batch_size = 1e-10, args.batchsize
                print "\nreg_mu: %f" % (mu_val)
                print "\nreg_lambda: %f" % (lambda_val)
                LG = Huber_One_Weight_Logistic_Regression(mu_val, lambda_val, learning_rate, max_iter, eps, batch_size)
                LG.fit(xTrain, yTrain, (args.sparsify==1), gm_opt_method=-1, verbos=True)
                if not np.isnan(np.linalg.norm(LG.w)):
                    print "\n\nfinal accuracy: %.6f\t|\tfinal auc: %6f\t|\ttest loss: %6f\t|\tprob test loss: %6f" % (LG.accuracy(LG.predict(xTest, (args.sparsify==1)), yTest), \
                        LG.auroc(LG.predict_proba(xTest, (args.sparsify==1)), yTest), LG.loss(xTest, yTest, (args.sparsify==1)), LG.probloss(xTest, yvalsTest, (args.sparsify==1)))
                print LG

                # plt.hist(LG.w, bins=50, normed=1, color='g', alpha=0.75)
                # plt.show()
                done = time.time()
                do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
                print do
                elapsed = done - start
                print elapsed
                np.savetxt('weight-out/'+sys.argv[0][:-3]+'_w.out', LG.w, delimiter=',')


'''
>>> threshold
5.0
>>> w = np.array([3,4,5,6,7,8])
>>> w = w.astype(float)
>>> lambda_val
0.0001
>>> mu_val
0.001
>>> np.piecewise(w, [np.absolute(w) < threshold, np.absolute(w) >= threshold], [lambda w: 2*lambda_val*w, lambda w: mu_val*np.sign(w)])
array([ 0.0006,  0.0008,  0.001 ,  0.001 ,  0.001 ,  0.001 ])
>>>

>>> reg_mu = 0.01
>>> reg_lambda = 0.001
>>> threshold = (reg_mu)/(reg_lambda*2.0)
>>> threshold
5.0
>>> w_array = np.array([2., -3., 4., -5., -6., 7., -8.]).astype(float)
>>> w_array = w_array.reshape((-1,1))
>>> w_array.shape
(7, 1)
>>> np.piecewise(w_array, [np.absolute(w_array) < threshold, np.absolute(w_array) >= threshold], [lambda w_array: 2*reg_lambda*w_array, lambda w_array: reg_mu*np.sign(w_array)])
array([[ 0.004],
    [-0.006],
   [ 0.008],
   [-0.01 ],
   [-0.01 ],
   [ 0.01 ],
   [-0.01 ]])
>>>

'''
