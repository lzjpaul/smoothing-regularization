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
from logistic_regression import Logistic_Regression
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
    x, y = loadData(args.datapath, onehot=(args.onehot==1), sparsify=(args.sparsify==1))
    print "loadData x shape: ", x.shape
    n_folds = 5
    accuracy_df = pandas.DataFrame()
    auc_df = pandas.DataFrame()
    loss_df = pandas.DataFrame()
    for i, (train_index, test_index) in enumerate(StratifiedKFold(y.reshape(y.shape[0]), n_folds=n_folds)):
        # if i > 0:
        #     break
        print "subsample i: ", i
        reg_lambda = [1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1000.]
        for reg in reg_lambda:
            start = time.time()
            st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
            print st
            print "train_index: ", train_index
            print "test_index: ", test_index
            xTrain, yTrain, xTest, yTest = x[train_index], y[train_index], x[test_index], y[test_index]
            learning_rate, max_iter = math.pow(10, (-1 * args.wlr)), args.maxiter
            eps, batch_size = 1e-10, args.batchsize
            print "\nreg_lambda: %f" % (reg)
            LG = Logistic_Regression(reg, learning_rate, max_iter, eps, batch_size)
            LG.fit(xTrain, yTrain, xTest, yTest, (args.sparsify==1), gm_opt_method=-1, verbos=True)
            if not np.isnan(np.linalg.norm(LG.w)):
                print "\n\nfinal accuracy: %.6f\t|\tfinal auc: %6f\t|\ttest loss: %6f" % (LG.accuracy(LG.predict(xTest, (args.sparsify==1)), yTest), \
                                                               LG.auroc(LG.predict_proba(xTest, (args.sparsify==1)), yTest), LG.loss(xTest, yTest, (args.sparsify==1)))
                accuracy_df.loc[i, (str(reg))] = LG.accuracy(LG.predict(xTest, (args.sparsify==1)), yTest)
                auc_df.loc[i, (str(reg))] = LG.auroc(LG.predict_proba(xTest, (args.sparsify==1)), yTest)
                loss_df.loc[i, (str(reg))] = (LG.loss(xTest, yTest, (args.sparsify==1)) * xTest.shape[0])
            print LG

            # plt.hist(LG.w, bins=50, normed=1, color='g', alpha=0.75)
            # plt.show()
            done = time.time()
            do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
            print do
            elapsed = done - start
            print elapsed
            np.savetxt('weight-out/'+sys.argv[0][:-3]+'_w.out', LG.w, delimiter=',')

    print accuracy_df
    pandas.options.display.float_format = '{:,.6f}'.format
    print accuracy_df.values
    print "\n\naccuracy pandas results\n\n"
    print "mean: ", accuracy_df.mean().values
    print "std: ", accuracy_df.std().values
    print "max mean index: ", accuracy_df.mean().idxmax()
    print "max mean: ", accuracy_df.mean().max()
    print "max mean std: ", accuracy_df.std().loc[accuracy_df.mean().idxmax()]
    print("accuracy best parameter %0.6f (+/-%0.06f)"
                          % (accuracy_df.mean().max(), accuracy_df.std().loc[accuracy_df.mean().idxmax()]))
    # print "max mean std: ", result_df.loc['Std'].loc[result_df['Mean'].idxmax()]
    print "best each subsample: \n", accuracy_df.max(axis=1)
    print "best each subsample index: \n", accuracy_df.idxmax(axis=1)
    print "mean of each subsample best: ", accuracy_df.max(axis=1).mean()
    print "std of each subsample best: ", accuracy_df.max(axis=1).std()
    print("accuracy best subsample %0.6f (+/-%0.06f)"
                          % (accuracy_df.max(axis=1).mean(), accuracy_df.max(axis=1).std()))

    print auc_df
    pandas.options.display.float_format = '{:,.6f}'.format
    print auc_df.values
    print "\n\nauc pandas results\n\n"
    print "mean: ", auc_df.mean().values
    print "std: ", auc_df.std().values
    print "max mean index: ", auc_df.mean().idxmax()
    print "max mean: ", auc_df.mean().max()
    print "max mean std: ", auc_df.std().loc[auc_df.mean().idxmax()]
    print("auc best parameter %0.6f (+/-%0.06f)"
                          % (auc_df.mean().max(), auc_df.std().loc[auc_df.mean().idxmax()]))
    # print "max mean std: ", result_df.loc['Std'].loc[result_df['Mean'].idxmax()]
    print "best each subsample: \n", auc_df.max(axis=1)
    print "best each subsample index: \n", auc_df.idxmax(axis=1)
    print "mean of each subsample best: ", auc_df.max(axis=1).mean()
    print "std of each subsample best: ", auc_df.max(axis=1).std()
    print("auc best subsample %0.6f (+/-%0.06f)"
                          % (auc_df.max(axis=1).mean(), auc_df.max(axis=1).std()))

    print loss_df
    pandas.options.display.float_format = '{:,.6f}'.format
    print loss_df.values
    print "\n\nloss pandas results\n\n"
    print "mean: ", loss_df.mean().values
    print "std: ", loss_df.std().values
    print "min mean index: ", loss_df.mean().idxmin()
    print "min mean: ", loss_df.mean().min()
    print "min mean std: ", loss_df.std().loc[loss_df.mean().idxmin()]
    print("loss best parameter %0.6f (+/-%0.06f)"
                          % (loss_df.mean().min(), loss_df.std().loc[loss_df.mean().idxmin()]))
    # print "min mean std: ", result_df.loc['Std'].loc[result_df['Mean'].idxmin()]
    print "best each subsample: \n", loss_df.min(axis=1)
    print "best each subsample index: \n", loss_df.idxmin(axis=1)
    print "mean of each subsample best: ", loss_df.min(axis=1).mean()
    print "std of each subsample best: ", loss_df.min(axis=1).std()
    print("loss best subsample %0.6f (+/-%0.06f)"
                          % (loss_df.min(axis=1).mean(), loss_df.min(axis=1).std()))
