'''
Cai Shaofeng - 2017.2
Implementation of the Gaussian Mixture Prior Logistic Regression
'''

'''
hyper:
(1) lr decay
(2) threshold for train_loss
'''
import sys
from logistic_regression import Logistic_Regression
from data_loader import *
from scipy.stats import norm as gaussian
import argparse
import math

class GM_Logistic_Regression(Logistic_Regression):
    def __init__(self, hyperpara, gm_num, pi, reg_lambda, learning_rate=0.1, pi_r_learning_rate=0.1, reg_lambda_s_learning_rate=0.1, max_iter=1000, eps=1e-4, batch_size=-1, validation_perc=0.0):
        Logistic_Regression.__init__(self, reg_lambda, learning_rate, max_iter, eps, batch_size, validation_perc)
        self.a, self.b, self.alpha, self.gm_num, self.pi = hyperpara[0], hyperpara[1], hyperpara[2], gm_num, pi
        print "self.a, self.b, self.alpha, self.gm_num, self.pi: ", self.a, self.b, self.alpha, self.gm_num, self.pi
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
        xTrain, yTrain = xTrain[index: (index + self.batch_size)], yTrain[index: (index + self.batch_size)]

        mu = self.sigmoid(np.matmul(xTrain, self.w))
        # check here, data part grad, need normalization with train_num/batch_size here
        grad_w = (self.trainNum/self.batch_size)*np.matmul(xTrain.T, (mu - yTrain))
        # gaussian mixture reg term grad
        self.calcResponsibility()
        reg_grad_w = np.sum(self.responsibility*self.reg_lambda, axis=1).reshape(self.w[:-1].shape) * self.w[:-1]
        grad_w += np.vstack((reg_grad_w, np.array([0.0])))

        # update gm prior: pi, reg_lambda
        # 0: fixed, 1: GD, 2: EM
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
        delta_reg_lambda = np.sum((self.responsibility / (2.0 * self.reg_lambda) - self.responsibility * 0.5 * self.w[:-1] * self.w[:-1]), axis=0).reshape((1,-1))
        delta_reg_lambda += (self.a - 1) / (self.reg_lambda.astype(float)) - self.b
        delta_reg_lambda = -delta_reg_lambda
        delta_reg_lambda_s = delta_reg_lambda * self.reg_lambda
        self.reg_lambda_s -= self.reg_lambda_s_lr(epoch_num) * delta_reg_lambda_s
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
        self.pi_r -= self.pi_r_lr(epoch_num) * delta_pi_r
        if iter_num % 100 == 0:
            print "self.pi_r      , self.pi_r norm:       ", self.pi_r, np.linalg.norm(self.pi_r)
            print "lr * delta_pi_r, lr * delta_pi_r norm: ", (self.pi_r_lr(epoch_num) * delta_pi_r), np.linalg.norm(self.pi_r_lr(epoch_num) * delta_pi_r)
            print "\n"
        #update pi
        pi_r_exp = np.exp(self.pi_r)
        self.pi = pi_r_exp / np.sum(pi_r_exp)

    def update_GM_Prior_EM(self, epoch_num, iter_num):
        # update pi
        self.reg_lambda = (2 * (self.a - 1) + np.sum(self.responsibility, axis=0)) / (2 * self.b + np.sum(self.responsibility * np.square(self.w[:-1]), axis=0))

        # update reg_lambda
        self.pi = (np.sum(self.responsibility, axis=0) + self.alpha - 1) / (self.featureNum + self.gm_num * (self.alpha - 1))

        # print 'reg_lambda', self.reg_lambda
        # print 'pi:', self.pi

    # calc the resposibilities for pj(wi)
    def calcResponsibility(self):
        # responsibility normalized with pi
        responsibility = gaussian.pdf(self.w[:-1], loc=np.zeros(shape=(1, self.gm_num)), scale=1/np.sqrt(self.reg_lambda))*self.pi
        # responsibility normalized with summation(denominator)
        self.responsibility = responsibility/(np.sum(responsibility, axis=1).reshape(self.w[:-1].shape))


    # model parameter
    def __str__(self):
        return 'model config {\thyper: [a-%d,b-%d,alpha-%d] reg: %s, lr: %.6f, pi_r_lr: %.6f, reg_lambda_s_lr: %.6f, batch_size: %5d\t}' \
               % (self.a, self.b, self.alpha, self.reg_lambda, self.learning_rate, self.pi_r_learning_rate, self.reg_lambda_s_learning_rate, self.batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-wlr', type=int, help='weight learning_rate (to the power of 10)')
    parser.add_argument('-pirlr', type=int, help='pi_r learning_rate (to the power of 10)')
    parser.add_argument('-lambdaslr', type=int, help='lambda_s learning_rate (to the power of 10)')
    parser.add_argument('-maxiter', type=int, help='max_iter')
    parser.add_argument('-gmoptmethod', type=int, help='gm optimization method: 0-fixed, 1-GD, 2-EM')
    args = parser.parse_args()

    # load the simulation data
    xTrain, xTest, yTrain, yTest, w = loadData('simulator.pkl', trainPerc=0.7)


    # run gm_prior lg model
    learning_rate, pi_r_learning_rate, reg_lambda_s_learning_rate = math.pow(10, (-1 * args.wlr)), math.pow(10, (-1 * args.pirlr)), math.pow(10, (-1 * args.lambdaslr))
    max_iter = args.maxiter
    gm_opt_method = args.gmoptmethod
    gm_num, a, b, alpha = 3, 1, 10, 50
    #pi, reg_lambda,  eps, batch_size \
    #    = np.array([0.70, 0.05, 0.2, 0.05]), np.array([200, 200, 10, 1.25]), 1e-10, 500
    pi, reg_lambda,  eps, batch_size \
        = [1.0/gm_num for _ in range(gm_num)], [_*10+1 for _ in  range(gm_num)], 1e-10, 500
    LG = GM_Logistic_Regression(hyperpara=[a, b, alpha], gm_num=gm_num, pi=pi, reg_lambda=reg_lambda, learning_rate=learning_rate, \
                                pi_r_learning_rate=pi_r_learning_rate, reg_lambda_s_learning_rate=reg_lambda_s_learning_rate, max_iter=max_iter, eps=eps, batch_size=batch_size)
    LG.fit(xTrain, yTrain, w, gm_opt_method, verbos=True)
    print "\n\nfinal accuracy: %.6f" % (LG.accuracy(LG.predict(xTest), yTest))
    print LG
    # plt.hist(LG.w, bins=50, normed=1, color='g', alpha=0.75)
    # plt.show()
    np.savetxt('weight-out/'+sys.argv[0][:-3]+'_w.out', LG.w, delimiter=',')


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

