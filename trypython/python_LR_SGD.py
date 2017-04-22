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
from data_loader_URL_complete import *
import argparse
import math
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import sparse
from scipy.sparse import linalg
import datetime
import time
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model import LogisticRegression, SGDClassifier
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-datapath', type=str, help='the dataset path, not svm')
    parser.add_argument('-onehot', type=int, help='need onehot or not')
    parser.add_argument('-sparsify', type=int, help='need sparsify or not')
    args = parser.parse_args()

    # load the permutated data
    x, y = loadData(args.datapath, onehot=(args.onehot==1), sparsify=(args.sparsify==1))
    print "loadData x shape: ", x.shape
    n_folds = 5
    for i, (train_index, test_index) in enumerate(StratifiedKFold(y.reshape(y.shape[0]), n_folds=n_folds)):
        if i > 0:
            break
        start = time.time()
        st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
        print st
        print "train_index: ", train_index
        print "test_index: ", test_index
        xTrain, yTrain, xTest, yTest = x[train_index], y[train_index], x[test_index], y[test_index]
        yTrain = yTrain.reshape(yTrain.shape[0])
        yTest = yTest.reshape(yTest.shape[0])
        # plt.hist(LG.w, bins=50, normed=1, color='g', alpha=0.75)
        # plt.show()
        regularization_array = np.array([1e-30, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10., 100.])
        for regularization in regularization_array:
        # print "fitting model..."
            print "regularization: ", regularization
            model = SGDClassifier(loss='log', penalty='l2', alpha=regularization, l1_ratio=0.15, fit_intercept=True, n_iter=5, \
                                  shuffle=True, verbose=1, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='constant', \
                                  eta0=0.1, power_t=0.5, class_weight=None, warm_start=False, average=False)
            # model = LogisticRegression(penalty = 'l2',C = args.penalty)
        # print "finish fitting model"
            model.fit(xTrain, yTrain)
            modelPred = model.predict_proba(xTest)
            y_true, y_scores = yTest, modelPred[:, 1].astype(np.float)
            print "roc"
            print roc_auc_score(y_true, y_scores)
            print "accuracy"
            y_pred = y_scores > 0.5
            print accuracy_score(y_true, y_pred)
            done = time.time()
            do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
            # print do
            elapsed = done - start
            # print elapsed


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
