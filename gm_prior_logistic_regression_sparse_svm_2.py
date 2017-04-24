'''
Luo Zhaojing - 2017.3
Gaussian Mixture(GM) Prior Logistic Regression
sparse reference:
https://github.com/EdwardRaff/JSAT/blob/9b914c6015b1e8643af475c287ee350043b07fd6/JSAT/src/jsat/classifiers/linear/StochasticMultinomialLogisticRegression.java
'''

'''
hyper:
(1) lr decay
(2) threshold for train_loss
'''
'''
batch-up regularization gradient for sparse input
-- delta_w
-- update_GM_Prior_GD
-- update_GM_Prior_EM
hyper_param_update_frequency
'''
import sys
from logistic_regression import Logistic_Regression
from data_loader import *
from scipy.stats import norm as gaussian
import argparse
import math
from sklearn.cross_validation import StratifiedKFold, cross_val_score
import datetime
import time
import pandas
class GM_Logistic_Regression(Logistic_Regression):
    def __init__(self, hyperpara, gmm_update_frequency, gm_num, pi, reg_lambda, learning_rate=0.1, pi_r_learning_rate=0.1, reg_lambda_s_learning_rate=0.1, max_iter=1000, eps=1e-4, batch_size=-1, validation_perc=0.0):
        Logistic_Regression.__init__(self, reg_lambda, learning_rate, max_iter, eps, batch_size, validation_perc)
        self.a, self.b, self.alpha, self.gmm_update_frequency, self.gm_num, self.pi = \
        hyperpara[0], hyperpara[1], hyperpara[2], gmm_update_frequency, gm_num, pi
        print "self.a, self.b, self.alpha, self.gmm_update_frequency, self.gm_num, self.pi: ", \
                self.a, self.b, self.alpha, self.gmm_update_frequency, self.gm_num, self.pi
        self.pi, self.reg_lambda = np.reshape(np.array(pi), (1, gm_num)), np.reshape(np.array(reg_lambda), (1, gm_num))
        self.pi_r, self.reg_lambda_s = np.log(self.pi), np.log(self.reg_lambda)
        print "init self.reg_lambda: ", self.reg_lambda
        print "init self.pi: ", self.pi
        print "init self.reg_lambda_s: ", self.reg_lambda_s
        print "init self.pi_r: ", self.pi_r
        self.pi_r_learning_rate, self.reg_lambda_s_learning_rate = pi_r_learning_rate, reg_lambda_s_learning_rate
        print "init self.pi_r_learning_rate, self.reg_lambda_s_learning_rate: ", self.pi_r_learning_rate, self.reg_lambda_s_learning_rate

    def pi_r_lr(self, epoch):
        if epoch < 100:
            return self.pi_r_learning_rate
        elif epoch < 150:
            return self.pi_r_learning_rate / float(10)
        else:
            return self.pi_r_learning_rate / float(100)

    def reg_lambda_s_lr(self, epoch):
        if epoch < 100:
            return self.reg_lambda_s_learning_rate
        elif epoch < 150:
            return self.reg_lambda_s_learning_rate / float(10)
        else:
            return self.reg_lambda_s_learning_rate / float(100)


    # calc the delta w to update w, using gm_prior_sgd here, update pi, reg_lambda here
    def delta_w(self, xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method):
        grad_w = self.likelihood_grad(xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method)
        # gaussian mixture reg term grad
        # the dimemsions that have non-zero data_grad
        update_w_idx = np.nonzero(grad_w[:-1].reshape(grad_w[:-1].shape[0]))[0]
        update_w = self.w[update_w_idx]
        self.calcPartialResponsibility(update_w)
        # RegScaled of the non_zero_grad_w update
        reg_grad_update_w = np.sum(self.partialresponsibility*self.reg_lambda, axis=1).reshape(update_w.shape) \
                              * update_w * (iter_num - self.u[update_w_idx]).reshape(update_w.shape)
        reg_grad_w = np.zeros((self.featureNum+1, 1))
        reg_grad_w[update_w_idx] = reg_grad_update_w
        grad_w += reg_grad_w
        self.u[update_w_idx] = np.full((update_w_idx.shape[0]), iter_num, dtype=int)

        if iter_num < 100 or iter_num % self.gmm_update_frequency == 0:
            # update gm prior: pi, reg_lambda
            # 0: fixed, 1: GD, 2: EM
            self.calcResponsibility()
            if gm_opt_method == 0:
                pass
            elif gm_opt_method == 1:
                self.update_GM_Prior_GD(epoch_num, iter_num)
            elif gm_opt_method == 2:
                self.update_GM_Prior_EM(epoch_num, iter_num)
            else:
                print "invalid gm_opt_method"

        return -grad_w

    def update_GM_Prior_GD(self, epoch_num, iter_num):
        #update reg_lambda_s
        delta_reg_lambda = np.sum((self.responsibility / (2.0 * self.reg_lambda) - self.responsibility * 0.5 * np.square(self.w[:-1])), axis=0).reshape((1,-1))
        delta_reg_lambda += (self.a - 1) / (self.reg_lambda.astype(float)) - self.b
        delta_reg_lambda = -delta_reg_lambda
        delta_reg_lambda_s = delta_reg_lambda * self.reg_lambda
        self.reg_lambda_s -= (iter_num - self.gm_prior_u) * self.reg_lambda_s_lr(epoch_num) * delta_reg_lambda_s
        if iter_num % 100 == 0:
            print "self.reg_lambda_s      , self.reg_lambda_s norm: ", self.reg_lambda_s, np.linalg.norm(self.reg_lambda_s)
            print "lr * delta_reg_lambda_s, lr * delta_reg_lambda_s norm: ", (self.reg_lambda_s_lr(epoch_num) * delta_reg_lambda_s), np.linalg.norm(self.reg_lambda_s_lr(epoch_num) * delta_reg_lambda_s)
            print "\n"
        #update reg_lambda
        self.reg_lambda = np.exp(self.reg_lambda_s)

        #update pi_r
        delta_pi = np.sum(self.responsibility / self.pi.astype(float), axis=0) + (self.alpha - 1) / self.pi.astype(float)
        delta_pi = -delta_pi
        delta_pi_k_j_mat = np.array([[(int(j==k)*self.pi[0,j] - self.pi[0,j] *self.pi[0,k]) for j in range(self.gm_num)] for k in range(self.gm_num)])
        delta_pi_r = np.matmul(delta_pi, delta_pi_k_j_mat)
        self.pi_r -= (iter_num - self.gm_prior_u) * self.pi_r_lr(epoch_num) * delta_pi_r
        if iter_num % 100 == 0:
            print "self.pi_r      , self.pi_r norm:       ", self.pi_r, np.linalg.norm(self.pi_r)
            print "lr * delta_pi_r, lr * delta_pi_r norm: ", (self.pi_r_lr(epoch_num) * delta_pi_r), np.linalg.norm(self.pi_r_lr(epoch_num) * delta_pi_r)
            print "\n"
        #update pi
        pi_r_exp = np.exp(self.pi_r)
        self.pi = pi_r_exp / np.sum(pi_r_exp)

        self.gm_prior_u = iter_num

    def update_GM_Prior_EM(self, epoch_num, iter_num):
        # update pi
        self.reg_lambda = (2 * (self.a - 1) + np.sum(self.responsibility, axis=0)) / (2 * self.b + np.sum(self.responsibility * np.square(self.w[:-1]), axis=0))

        # update reg_lambda
        self.pi = (np.sum(self.responsibility, axis=0) + self.alpha - 1) / (self.featureNum + self.gm_num * (self.alpha - 1))

        self.gm_prior_u = iter_num
        # print 'reg_lambda', self.reg_lambda
        # print 'pi:', self.pi

    # calc the resposibilities for pj(wi)
    def calcResponsibility(self):
        # responsibility normalized with pi
        responsibility = gaussian.pdf(self.w[:-1], loc=np.zeros(shape=(1, self.gm_num)), scale=1/np.sqrt(self.reg_lambda))*self.pi
        # responsibility normalized with summation(denominator)
        self.responsibility = responsibility/(np.sum(responsibility, axis=1).reshape(self.w[:-1].shape))

    # calc the partial resposibilities for partial non-zero likelihood dimensions pj(wi)
    def calcPartialResponsibility(self, update_w):
        # responsibility normalized with pi
        partialresponsibility = gaussian.pdf(update_w, loc=np.zeros(shape=(1, self.gm_num)), scale=1/np.sqrt(self.reg_lambda))*self.pi
        # responsibility normalized with summation(denominator)
        self.partialresponsibility = partialresponsibility/(np.sum(partialresponsibility, axis=1).reshape(update_w.shape))

    # w loss
    def w_loss(self):
        responsibility = gaussian.pdf(self.w[:-1], loc=np.zeros(shape=(1, self.gm_num)), scale=1/np.sqrt(self.reg_lambda))*self.pi
        responsibility_w = np.sum(responsibility, axis=1)
        log_responsibility_w = -np.log(responsibility_w)
        return np.sum(log_responsibility_w)

    # model parameter
    def __str__(self):
        return 'model config {\thyper: [a-%f,b-%f,alpha-%d] reg: %s, pi: %s, lr: %.6f, pi_r_lr: %.6f, reg_lambda_s_lr: %.6f, batch_size: %5d\t}' \
               % (self.a, self.b, self.alpha, self.reg_lambda, self.pi, self.learning_rate, self.pi_r_learning_rate, self.reg_lambda_s_learning_rate, self.batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-datapath', type=str, help='the dataset path, not svm')
    parser.add_argument('-onehot', type=int, help='need onehot or not')
    parser.add_argument('-sparsify', type=int, help='need sparsify or not')
    parser.add_argument('-batchsize', type=int, help='batchsize')
    parser.add_argument('-wlr', type=int, help='weight learning_rate (to the power of 10)')
    parser.add_argument('-pirlr', type=int, help='pi_r learning_rate (to the power of 10)')
    parser.add_argument('-lambdaslr', type=int, help='lambda_s learning_rate (to the power of 10)')
    parser.add_argument('-maxiter', type=int, help='max_iter')
    parser.add_argument('-gmmuptfreq', type=int, help='gmm update frequency')
    parser.add_argument('-gmnum', type=int, help='gm_number')
    parser.add_argument('-gmoptmethod', type=int, help='gm optimization method: 0-fixed, 1-GD, 2-EM')
    args = parser.parse_args()

    # load the permutated data
    x, y = loadData(args.datapath, onehot=(args.onehot==1), sparsify=(args.sparsify==1))
    print "loadData x shape: ", x.shape
    n_folds = 5
    accuracy_df = pandas.DataFrame()
    auc_df = pandas.DataFrame()
    loss_df = pandas.DataFrame()
    for i, (train_index, test_index) in enumerate(StratifiedKFold(y.reshape(y.shape[0]), n_folds=n_folds)):
        if i > 0:
            break
        a, b, alpha = [1., 10.], [100.], [int(np.sqrt(x.shape[1]))]
        for alpha_val in alpha:
            for a_val in a:
                for b_val in b:
                    start = time.time()
                    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
                    print st
                    print "train_index: ", train_index
                    print "test_index: ", test_index
                    xTrain, yTrain, xTest, yTest = x[train_index], y[train_index], x[test_index], y[test_index]
                    # run gm_prior lg model
                    learning_rate, pi_r_learning_rate, reg_lambda_s_learning_rate = math.pow(10, (-1 * args.wlr)), math.pow(10, (-1 * args.pirlr)), math.pow(10, (-1 * args.lambdaslr))
                    max_iter = args.maxiter
                    gm_opt_method = args.gmoptmethod
                    gmm_update_frequency, gm_num = args.gmmuptfreq, args.gmnum
                    pi, reg_lambda,  eps, batch_size \
                        = [1.0/gm_num for _ in range(gm_num)], [_*10+1 for _ in  range(gm_num)], 1e-10, args.batchsize
                    LG = GM_Logistic_Regression(hyperpara=[a_val, b_val, alpha_val], gmm_update_frequency=gmm_update_frequency, gm_num=gm_num, pi=pi, reg_lambda=reg_lambda, learning_rate=learning_rate, \
                                                pi_r_learning_rate=pi_r_learning_rate, reg_lambda_s_learning_rate=reg_lambda_s_learning_rate, max_iter=max_iter, eps=eps, batch_size=batch_size)
                    LG.fit(xTrain, yTrain, xTest, yTest, (args.sparsify==1), gm_opt_method=gm_opt_method, verbos=True)
                    if not np.isnan(np.linalg.norm(LG.w)):
                        print "\n\nfinal accuracy: %.6f\t|\tfinal auc: %6f\t|\ttest loss: %6f" % (LG.accuracy(LG.predict(xTest, (args.sparsify==1)), yTest), \
                                                               LG.auroc(LG.predict_proba(xTest, (args.sparsify==1)), yTest), LG.loss(xTest, yTest, (args.sparsify==1)))
                        accuracy_df.loc[i, (str(alpha_val) + ',' + str(a_val) + ',' + str(b_val))] = LG.accuracy(LG.predict(xTest, (args.sparsify==1)), yTest)
                        auc_df.loc[i, (str(alpha_val) + ',' + str(a_val) + ',' + str(b_val))] = LG.auroc(LG.predict_proba(xTest, (args.sparsify==1)), yTest)
                        loss_df.loc[i, (str(alpha_val) + ',' + str(a_val) + ',' + str(b_val))] = (LG.loss(xTest, yTest, (args.sparsify==1)) * xTest.shape[0])
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

# command python gm_prior_logistic_regression.py -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 30000 -gmnum 4 -a 0 -b 1 -alpha 50 -gmoptmethod 1
'''
final accuracy: 0.839333
model config {	hyper: [a-1,b-10,alpha-50] reg: [[1000  100   50   10]], lr: 0.000010, batch_size:   500, best_iter:   3100, best_accuracy: 0.823333	}

final accuracy: 0.846000
model config {	hyper: [a-1,b-10,alpha-50] reg: [[100  10   5   1]], lr: 0.000010, batch_size:   500, best_iter:   2000, best_accuracy: 0.820952	} [[-0.03937882]

final accuracy: 0.832333
model config {	hyper: [a-1,b-10,alpha-50] reg: [[ 10.    1.    0.5   0.1]], lr: 0.000010, batch_size:   500, best_iter:   1200, best_accuracy: 0.818571	}


# print 'reg', np.sum(self.responsibility*self.reg_lambda, axis=1).reshape(self.w.shape)
# print 'w', self.w
# print 'reg_w', np.sum(self.responsibility*self.reg_lambda, axis=1).reshape(self.w.shape) * self.w
# print 'grad_w', grad_w
'''

'''
    def update_GM_Prior(self):
        # update pi
        grad_pi = np.sum(self.responsibility, axis=0) / self.pi
        #grad_pi += (self.alpha - 1) / self.pi
        pi_k_j = np.array([[(j == k) * self.pi[0][j] - self.pi[0][j] * self.pi[0][k] for j in xrange(self.gm_num)] for k in xrange(self.gm_num)])
        grad_pi_r = np.dot(grad_pi, pi_k_j)

        self.pi_r -= 0.001 * grad_pi_r
        self.pi = self.softmax(self.pi_r)

        print 'pi_r', self.pi_r
        print 'grad_pi_r', grad_pi_r
        print 'pi', self.pi

        # update reg_lambda
        grad_lambda = (np.sum(self.responsibility, axis=0)/(self.reg_lambda) - np.sum(self.responsibility*np.square(self.w), axis=0))/2
        #grad_lambda += (self.a-1)/self.reg_lambda - self.b
        #grad_lambda_s = grad_lambda*self.reg_lambda


        #self.reg_lambda = self.reg_lambda - 0.001 * grad_lambda

        # print 'grad_lambda', grad_lambda
        # print 'reg_lambda', self.reg_lambda
        #print 'lambda_s', self.reg_lambda_s
        #print 'grad_lambda_s', grad_lambda_s

        #self.reg_lambda_s -= 0.000001 * grad_lambda_s
        #self.reg_lambda = np.exp(self.reg_lambda_s)
'''

'''
def update_GM_Prior(self):
    # update pi
    self.reg_lambda = (2 * (self.a - 1) + np.sum(self.responsibility, axis=0)) / (
    2 * self.b + np.sum(self.reg_lambda * np.square(self.w), axis=0))

    # update reg_lambda
    self.pi = (np.sum(self.responsibility, axis=0) + self.alpha - 1) / (
    self.featureNum + self.gm_num * (self.alpha - 1))

    # print 'reg_lambda', self.reg_lambda
    # print 'pi:', self.pi
'''

'''
def softmax(self, x):
    #Compute softmax values for each sets of scores in x.
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)
'''

