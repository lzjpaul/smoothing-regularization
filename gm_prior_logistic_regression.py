'''
Cai Shaofeng - 2017.2
Implementation of the Gaussian Mixture Prior Logistic Regression
'''

from logistic_regression import Logistic_Regression
from data_loader import *
from scipy.stats import norm as gaussian

class GM_Logistic_Regression(Logistic_Regression):
    def __init__(self, hyperpara, gm_num, pi, reg_lambda, learning_rate=0.1, max_iter=1000, eps=1e-4, batch_size=-1, validation_perc=0.3):
        Logistic_Regression.__init__(self, reg_lambda, learning_rate, max_iter, eps, batch_size, validation_perc)
        self.a, self.b, self.alpha, self.gm_num, self.pi = hyperpara[0], hyperpara[1], hyperpara[2], gm_num, pi
        self.pi, self.reg_lambda = np.reshape(np.array(pi), (1, gm_num)), np.reshape(np.array(reg_lambda), (1, gm_num))

    # calc the delta w to update w, using gm_prior_sgd here, update pi, reg_lambda here
    def delta_w(self, xTrain, yTrain):
        # mini batch, not used here
        if self.batch_size != -1:
            randomIndex = np.random.random_integers(0, xTrain.shape[0] - 1, self.batch_size)
            xTrain, yTrain = xTrain[randomIndex], yTrain[randomIndex]

        mu = self.sigmoid(np.matmul(xTrain, self.w))
        # check here, data part grad, need normalization with train_num/batch_size here
        grad_w = (self.trainNum/self.batch_size)*np.matmul(xTrain.T, (mu - yTrain))
        # gaussian mixture reg term grad
        self.calcResponsibility()
        # print 'reg', np.sum(self.responsibility*self.reg_lambda, axis=1).reshape(self.w.shape)
        # print 'w', self.w
        # print 'reg_w', np.sum(self.responsibility*self.reg_lambda, axis=1).reshape(self.w.shape) * self.w
        # print 'grad_w', grad_w
        grad_w += np.sum(self.responsibility*self.reg_lambda, axis=1).reshape(self.w.shape) * self.w
        return -grad_w

    # calc the resposibilities for pj(wi)
    def calcResponsibility(self):
        # responsibility normalized with pi
        responsibility = gaussian.pdf(self.w, loc=np.zeros(shape=(1, self.gm_num)), scale=1/np.sqrt(self.reg_lambda))*self.pi
        # responsibility normalized with summation(denominator)
        self.responsibility = responsibility/(np.sum(responsibility, axis=1).reshape(self.w.shape))

    # model parameter
    def __str__(self):
        return 'model config {\thyper: [a-%d,b-%d,alpha-%d] reg: %s, lr: %.6f, batch_size: %5d, best_iter: %6d, best_accuracy: %.6f\t}' \
               % (self.a, self.b, self.alpha, self.reg_lambda, self.learning_rate, self.batch_size, self.best_iter, self.best_accuracy)

if __name__ == '__main__':
    # load the simulation data
    xTrain, xTest, yTrain, yTest = loadData('simulator.pkl', trainPerc=0.7)

    # run gm_prior lg model
    a, b, alpha = 1, 10, 50
    pi, reg_lambda, learning_rate, max_iter, eps, batch_size = [0.4, 0.3, 0.2, 0.1], [10, 1, .50, .10], 0.00001, 5000, 1e-3, 500
    LG = GM_Logistic_Regression(hyperpara=[a, b, alpha], gm_num=4, pi=pi, reg_lambda=reg_lambda, learning_rate=learning_rate, max_iter=max_iter, eps=eps, batch_size=batch_size)
    LG.fit(xTrain, yTrain)
    print "\n\nfinal accuracy: %.6f" % (LG.accuracy(LG.predict(xTest), yTest))
    print LG, LG.best_w



'''
final accuracy: 0.839333
model config {	hyper: [a-1,b-10,alpha-50] reg: [[1000  100   50   10]], lr: 0.000010, batch_size:   500, best_iter:   3100, best_accuracy: 0.823333	}

final accuracy: 0.846000
model config {	hyper: [a-1,b-10,alpha-50] reg: [[100  10   5   1]], lr: 0.000010, batch_size:   500, best_iter:   2000, best_accuracy: 0.820952	} [[-0.03937882]

final accuracy: 0.832333
model config {	hyper: [a-1,b-10,alpha-50] reg: [[ 10.    1.    0.5   0.1]], lr: 0.000010, batch_size:   500, best_iter:   1200, best_accuracy: 0.818571	}
'''
