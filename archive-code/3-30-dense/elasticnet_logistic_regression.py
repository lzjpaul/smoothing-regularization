'''
Luo Zhaojing - 2017.3
ElasticNet Logistic Regression
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
# base logistic regression class
class ElasticNet_Logistic_Regression(Logistic_Regression):
    def __init__(self, l1_ratio, reg_lambda=1, learning_rate=0.1, max_iter=1000, eps=1e-4, batch_size=-1, validation_perc=0.0):
        Logistic_Regression.__init__(self, reg_lambda, learning_rate, max_iter, eps, batch_size, validation_perc)
        self.l1_ratio = l1_ratio
        print "self.l1_ratio: ", self.l1_ratio

    # calc the delta w to update w, using sgd here
    def delta_w(self, xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method):
        grad_w = self.likelihood_grad(xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method)
        reg_grad_w = self.reg_lambda * self.l1_ratio * np.sign(self.w) + self.reg_lambda * (1 - self.l1_ratio) * self.w
        reg_grad_w[-1, 0] = 0.0 # bias
        grad_w += reg_grad_w
        return -grad_w

    # model parameter
    def __str__(self):
        return 'model config {\tl1_ratio: %.6f, reg: %.6f, lr: %.6f, batch_size: %5d\t}' \
            % (self.l1_ratio, self.reg_lambda, self.learning_rate, self.batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-datapath', type=str, help='the dataset path, not svm')
    parser.add_argument('-onehot', type=int, help='need onehot or not')
    parser.add_argument('-sparsify', type=int, help='need sparsify or not')
    parser.add_argument('-batchsize', type=int, help='batchsize')
    parser.add_argument('-wlr', type=int, help='weight learning_rate (to the power of 10)')
    parser.add_argument('-maxiter', type=int, help='max_iter')
    args = parser.parse_args()

    # load the simulation data
    x, y = loadData(args.datapath, onehot=(args.onehot==1), sparsify=(args.sparsify==1))
    n_folds = 5
    for i, (train_index, test_index) in enumerate(StratifiedKFold(y.reshape(y.shape[0]), n_folds=n_folds)):
        if i > 0:
            break
        start = time.time()
        st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
        print st
        xTrain, yTrain, xTest, yTest = x[train_index], y[train_index], x[test_index], y[test_index]
        learning_rate, max_iter = math.pow(10, (-1 * args.wlr)), args.maxiter
        l1_ratio, reg_lambda, eps, batch_size = 0.5, 10, 1e-10, args.batchsize
        print "\nreg_lambda: %f" % (reg_lambda)
        LG = ElasticNet_Logistic_Regression(l1_ratio, reg_lambda, learning_rate, max_iter, eps, batch_size)
        LG.fit(xTrain, yTrain, gm_opt_method=-1, verbos=True)
        print "\n\nfinal accuracy: %.6f\t|\tfinal auc: %6f" % (LG.accuracy(LG.predict(xTest), yTest), LG.auroc(LG.predict_proba(xTest), yTest))
        print LG

        # plt.hist(LG.w, bins=50, normed=1, color='g', alpha=0.75)
        # plt.show()
        done = time.time()
        do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
        print do
        elapsed = done - start
        print elapsed
        np.savetxt('weight-out/'+sys.argv[0][:-3]+'_w.out', LG.w, delimiter=',')


    # train_accuracy, test_accuracy = [], []
    # #create logistic regression class
    # reg_lambda, learning_rate, max_iter, eps, batch_size = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 0.00001, 5000, 1e-3, 500
    # for reg in reg_lambda:
    #     print "\nreg_lambda: %f" %(reg)
    #     LG = Logistic_Regression(reg, learning_rate, max_iter, eps, batch_size)
    #     LG.fit(xTrain, yTrain)
    #     test_accuracy.append(LG.accuracy(LG.predict(xTest), yTest))
    #     print "finally accuracy: %.6f" %(test_accuracy[-1])
    #     print LG
    #
    # fig, ax = plt.subplots()
    # ax.plot(reg_lambda, train_accuracy, 'r-', label='train', ); ax.plot(reg_lambda, test_accuracy, 'b-', label='test')
    # ax.set_xscale('log'); ax.set_xticks(reg_lambda); plt.xlabel('reg_lambda');plt.ylabel('accuracy');plt.legend(loc='upper left')
    # plt.title('accuracy VS reg_lambda'); #plt.savefig('data/l2_accuracy.eps', format='eps', dpi=1000)
    # plt.show()



'''

reg_lambda: 0.000100
finally accuracy: 0.836000
model config {	reg: 0.000100, lr: 0.000010, batch_size:   500, best_iter:   1300, best_accuracy: 0.818571	}

reg_lambda: 0.001000
finally accuracy: 0.840000
model config {	reg: 0.001000, lr: 0.000010, batch_size:   500, best_iter:    600, best_accuracy: 0.821905	}

reg_lambda: 0.010000
finally accuracy: 0.840667
model config {	reg: 0.010000, lr: 0.000010, batch_size:   500, best_iter:    700, best_accuracy: 0.818571	}

reg_lambda: 0.100000
finally accuracy: 0.842333
model config {	reg: 0.100000, lr: 0.000010, batch_size:   500, best_iter:   1200, best_accuracy: 0.818571	}

reg_lambda: 1.000000
finally accuracy: 0.844000
model config {	reg: 1.000000, lr: 0.000010, batch_size:   500, best_iter:   1000, best_accuracy: 0.822381	}

reg_lambda: 10.000000
finally accuracy: 0.841333
model config {	reg: 10.000000, lr: 0.000010, batch_size:   500, best_iter:   1000, best_accuracy: 0.820952	}

reg_lambda: 100.000000
finally accuracy: 0.841333
model config {	reg: 100.000000, lr: 0.000010, batch_size:   500, best_iter:   2400, best_accuracy: 0.818571	}

reg_lambda: 1000.000000
finally accuracy: 0.805667
model config {	reg: 1000.000000, lr: 0.000010, batch_size:   500, best_iter:   2100, best_accuracy: 0.788571	}


for test use
# print 'reg', self.reg_lambda
# print 'w', self.w
# print 'reg_w', self.reg_lambda * self.w
# print 'grad_w', grad_w

# reg_lambda, learning_rate, max_iter, eps, batch_size = 1, 0.00001, 3000, 1e-3, 500
# print "\nreg_lambda: %f" % (reg_lambda)
# LG = Logistic_Regression(reg_lambda, learning_rate, max_iter, eps, batch_size)
# LG.fit(xTrain, yTrain)
# print "\n\nfinal accuracy: %.6f" % (LG.accuracy(LG.predict(xTest), yTest))
# print LG, LG.best_w
'''
