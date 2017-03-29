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

# base logistic regression class
class Logistic_Regression(object):
    def __init__(self, reg_lambda=1, learning_rate=0.1, max_iter=1000, eps=1e-4, batch_size=-1, validation_perc=0.0):
        self.reg_lambda, self.learning_rate, self.max_iter = reg_lambda, learning_rate, max_iter
        self.eps, self.batch_size, self.validation_perc = eps, batch_size, validation_perc
        print "self.reg_lambda, self.learning_rate, self.max_iter: ", self.reg_lambda, self.learning_rate, self.max_iter
        print "self.eps, self.batch_size, self.validation_perc: ", self.eps, self.batch_size, self.validation_perc

    def w_lr(self, epoch):
        if epoch < 100:
            return self.learning_rate
        elif epoch < 150:
            return self.learning_rate / float(10)
        else:
            return self.learning_rate / float(100)

    def likelihood_grad(self, xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method):
        xTrain, yTrain = xTrain[index : (index + self.batch_size)], yTrain[index : (index + self.batch_size)]
        print "index: ", index
        # print "xTrain[5,:100]: ", np.linalg.norm(xTrain[5])
        # print "xTrain[5, :100]: ", xTrain[5, :100]
        # print "yTrain[10:20]: ", yTrain[10:20]
        mu = self.sigmoid(np.matmul(xTrain, self.w))
        # print "mu[10:20]: ", mu[10:20]
        # check here, no regularization over bias term # need normalization with xTrain.shape[0]/batch_size here
        grad_w = (1.0/self.batch_size)*np.matmul(xTrain.T, (mu - yTrain))
        print "data grad: ", np.linalg.norm(grad_w)

        return grad_w
    # calc the delta w to update w, using sgd here
    def delta_w(self, xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method):
        grad_w = self.likelihood_grad(xTrain, yTrain, index, epoch_num, iter_num, gm_opt_method)
        reg_grad_w = self.reg_lambda * self.w
        reg_grad_w[-1, 0] = 0.0 # bias
        grad_w += reg_grad_w
        print "regularization grad: ", np.linalg.norm(reg_grad_w)
        return -grad_w


    def fit(self, xTrain, yTrain, ishuber=False, gm_opt_method=-1, verbos=False):
        np.random.seed(10)
        # find the number of class and feature, allocate memory for model parameters
        self.trainNum, self.featureNum = xTrain.shape[0], xTrain.shape[1]
        if ishuber:
            self.w1 = np.random.normal(0, 0.1, size=(self.featureNum+1, 1))#np.zeros(shape=(self.featureNum+1, 1), dtype='float32')
            self.w2 = np.random.normal(0, 0.1, size=(self.featureNum+1, 1))#np.zeros(shape=(self.featureNum+1, 1), dtype='float32')
            self.w = np.add(self.w1, self.w2)
        else:
            self.w = np.random.normal(0, 0.1, size=(self.featureNum+1, 1))#np.zeros(shape=(self.featureNum+1, 1), dtype='float32')
        print "self.w[:10]: ", self.w[:10]
        print "self.w1[:10]: ", self.w1[:10]
        print "self.w2[:10]: ", self.w2[:10]
        # adding 1s to each training examples
        xTrain = np.hstack((xTrain, np.ones(shape=(self.trainNum, 1))))

        # validation set
        validationNum = int(self.validation_perc*xTrain.shape[0])
        xVallidation, yVallidation = xTrain[:validationNum, ], yTrain[:validationNum, ]
        xTrain, yTrain = xTrain[validationNum:, ], yTrain[validationNum:, ]
        idx = np.random.permutation(xTrain.shape[0])
        print "data idx: ", idx
        xTrain = xTrain[idx]
        yTrain = yTrain[idx]
        try:
            iter, pre_train_loss = 0, 0.0
            # minibatch initialization
            batch_iter = 0
            np.random.seed(10)
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
                    print "self.w.norm: ", np.linalg.norm(self.w)
                    # calc the delta_w2 to update w2
                    delta_w2 = self.delta_w2(xTrain, yTrain, index, epoch_num, iter, gm_opt_method)
                    # update w2
                    self.w2 += self.w_lr(epoch_num) * delta_w2
                    self.w = np.add(self.w1, self.w2)
                    print "self.w.norm: ", np.linalg.norm(self.w)
                else:
                    # calc the delta_w to update w
                    delta_w = self.delta_w(xTrain, yTrain, index, epoch_num, iter, gm_opt_method)
                    # update w
                    self.w += self.w_lr(epoch_num) * delta_w

                # stop updating if converge
                # https://www.coursera.org/learn/machine-learning/lecture/fKi0M/stochastic-gradient-descent-convergence
                iter += 1
                batch_iter += 1
                if iter % 100 == 0:
                    train_loss = self.loss(xTrain, yTrain)
                    print "train_loss %10.10f abs(train_loss - pre_train_loss) %10.10f self.eps %10.10f"%(train_loss, abs(train_loss - pre_train_loss), self.eps)
                    print "self.max_iter: ", self.max_iter
                    if iter > self.max_iter or abs(train_loss - pre_train_loss) < self.eps:
                        break
                    pre_train_loss = train_loss
                # print useful information
                if iter % 100 == 0:
                    # print np.sum(np.abs(self.w))/self.featureNum, np.linalg.norm(self.w, ord=2)
                    train_accuracy = self.accuracy(self.predict(xTrain), yTrain)
                    train_loss = self.loss(xTrain, yTrain)
                    if verbos:
                        print "iter %4d\t|\ttrain_accuracy %10.6f\t|\ttrain_loss %10.10f"%(iter, train_accuracy, train_loss)
                        if hasattr(self, 'pi'):
                            regularization_loss = self.w_loss()
                            print "w norm %10.6f\t|\tdelta_w norm %10.6f\t|\tw_loss %10.10f"%(np.linalg.norm(self.w, ord=2), np.linalg.norm(self.w_lr(epoch_num) * delta_w, ord=2), regularization_loss)
                            print "lr %8.6f\t|\toverall loss %10.10f"%(self.w_lr(epoch_num), (train_loss+regularization_loss))
                            print "pi, reg_lambda: ", self.pi, self.reg_lambda
                            print "lr, pi_r_l, reg_lambda_s_lr: ",self.w_lr(epoch_num), self.pi_r_lr(epoch_num), self.reg_lambda_s_lr(epoch_num)
                            print "\n"
        finally:
            self.w = self.w

    # loss function
    def loss(self, samples, yTrue):
        threshold = 1e-320
        yTrue = yTrue.astype(int)
        mu = self.sigmoid(np.matmul(samples, self.w))
        mu_false = (1-mu)
        return np.sum((-yTrue * np.log(np.piecewise(mu, [mu < threshold, mu >= threshold], [threshold, lambda mu:mu])) \
                       - (1-yTrue) * np.log(np.piecewise(mu_false, [mu_false < threshold, mu_false >= threshold], [threshold, lambda mu_false:mu_false]))), axis = 0) / float(samples.shape[0])

    # predict result
    def predict(self, samples):
        if samples.shape[1] != self.w.shape[0]:
            samples = np.hstack((samples, np.ones(shape=(samples.shape[0], 1))))
        return np.matmul(samples, self.w)>0.0

    # predict probability
    def predict_proba(self, samples):
        if samples.shape[1] != self.w.shape[0]:
            samples = np.hstack((samples, np.ones(shape=(samples.shape[0], 1))))
        return self.sigmoid(np.matmul(samples, self.w))

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
        xTrain, yTrain, xTest, yTest = x[train_index], y[train_index], x[test_index], y[test_index]
        learning_rate, max_iter = math.pow(10, (-1 * args.wlr)), args.maxiter
        reg_lambda, eps, batch_size = 0.1, 1e-10, args.batchsize
        print "\nreg_lambda: %f" % (reg_lambda)
        LG = Logistic_Regression(reg_lambda, learning_rate, max_iter, eps, batch_size)
        LG.fit(xTrain, yTrain, gm_opt_method=-1, verbos=True)
        print "\n\nfinal accuracy: %.6f\t|\tfinal auc: %6f" % (LG.accuracy(LG.predict(xTest), yTest), LG.auroc(LG.predict_proba(xTest), yTest))
        print LG

        # plt.hist(LG.w, bins=50, normed=1, color='g', alpha=0.75)
        # plt.show()
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
