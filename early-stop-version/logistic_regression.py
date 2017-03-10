'''
Cai Shaofeng - 2017.2
Implementation of the Logistic Regression
'''

from data_loader import *
# base logistic regression class
class Logistic_Regression(object):
    def __init__(self, reg_lambda=1, learning_rate=0.1, max_iter=1000, eps=1e-4, batch_size=-1, validation_perc=0.3):
        self.reg_lambda, self.learning_rate, self.max_iter = reg_lambda, learning_rate, max_iter
        self.eps, self.batch_size, self.validation_perc = eps, batch_size, validation_perc

    # calc the delta w to update w, using sgd here
    def delta_w(self, xTrain, yTrain, index):
        xTrain, yTrain = xTrain[index : (index + self.batch_size)], yTrain[index : (index + self.batch_size)]

        mu = self.sigmoid(np.matmul(xTrain, self.w))
        # check here, no regularization over bias term # need normalization with xTrain.shape[0]/batch_size here
        grad_w = (self.trainNum/self.batch_size)*np.matmul(xTrain.T, (mu - yTrain))
        reg_grad_w = self.reg_lambda * self.w
        reg_grad_w[-1, 0] = 0.0 # bias
        grad_w += reg_grad_w
        return -grad_w

    def fit(self, xTrain, yTrain, verbos=False):
        # find the number of class and feature, allocate memory for model parameters
        self.trainNum, self.featureNum = xTrain.shape[0], xTrain.shape[1]
        self.w = np.random.normal(0, 0.01, size=(self.featureNum+1, 1))#np.zeros(shape=(self.featureNum+1, 1), dtype='float32')

        # adding 1s to each training examples
        xTrain = np.hstack((np.ones(shape=(self.trainNum, 1)), xTrain))

        # validation set
        validationNum = int(self.validation_perc*xTrain.shape[0])
        xVallidation, yVallidation = xTrain[:validationNum, ], yTrain[:validationNum, ]
        xTrain, yTrain = xTrain[validationNum:, ], yTrain[validationNum:, ]


        try:
            iter, self.best_accuracy, self.best_iter = 0, 0.0, 0
            # minibatch initialization
            batch_iter = 0
            np.random.seed(10)
            idx = np.random.permutation(xTrain.shape[0])
            xTrain = xTrain[idx]
            yTrain = yTrain[idx]
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

                # calc the delta_w to update w
                delta_w = self.delta_w(xTrain, yTrain, index)
                # update w
                self.w += self.learning_rate * delta_w

                # stop updating if converge
                iter += 1
                batch_iter += 1
                if iter > self.max_iter or np.linalg.norm(delta_w, ord=2) < self.eps:
                    break
                if iter % 100 == 0:
                    # print np.sum(np.abs(self.w))/self.featureNum, np.linalg.norm(self.w, ord=2)
                    test_accuracy, train_accuracy = self.accuracy(self.predict(xVallidation), yVallidation), self.accuracy(self.predict(xTrain), yTrain)
                    test_loss, train_loss = self.loss(xVallidation, yVallidation), self.loss(xTrain, yTrain)
                    if self.best_accuracy < test_accuracy:
                         self.best_w, self.best_accuracy, self.best_iter = np.copy(self.w), test_accuracy, iter
                    if verbos:
                        print "iter %4d\t|\ttrain_accuracy %10.6f\t|\ttest_accuracy %10.6f\t|\tbest_accuracy %10.6f"\
                          %(iter, train_accuracy, test_accuracy, self.best_accuracy)
                        print "iter %4d\t|\ttrain_loss     %10.6f\t|\ttest_loss     %10.6f\t" \
                              % (iter, train_loss, test_loss)
                        print "w norm %10.6f\t|\tdelta_w norm %10.6f "%(np.linalg.norm(self.w, ord=2), np.linalg.norm(self.learning_rate * delta_w, ord=2))
                        if hasattr(self, 'pi'):
                            print "pi, reg_lambda: ", self.pi, self.reg_lambda

        # except:
        #     pass
        finally:
            self.w = self.best_w

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
            samples = np.hstack((np.ones(shape=(samples.shape[0], 1)), samples))
        return np.matmul(samples, self.w)>0.0

    # calc accuracy
    def accuracy(self, yPredict, yTrue):
        return np.sum(yPredict == yTrue) / float(yTrue.shape[0])

    # sigmoid function
    def sigmoid(self, matrix):
        return 1.0/(1.0+np.exp(-matrix))

    # model parameter
    def __str__(self):
        return 'model config {\treg: %.6f, lr: %.6f, batch_size: %5d, best_iter: %6d, best_accuracy: %.6f\t}' \
            % (self.reg_lambda, self.learning_rate, self.batch_size, self.best_iter, self.best_accuracy)

if __name__ == '__main__':
    # load the simulation data
    xTrain, xTest, yTrain, yTest = loadData('simulator.pkl', trainPerc=0.7)
    
    reg_lambda, learning_rate, max_iter, eps, batch_size = 0.0, 0.00001, 50000, 1e-4, 500
    print "\nreg_lambda: %f" % (reg_lambda)
    LG = Logistic_Regression(reg_lambda, learning_rate, max_iter, eps, batch_size)
    LG.fit(xTrain, yTrain, verbos=True)
    print "\n\nfinal accuracy: %.6f" % (LG.accuracy(LG.predict(xTest), yTest))
    print LG, LG.best_w
    
    plt.hist(LG.w, bins=50, normed=1, color='g', alpha=0.75)
    plt.show()

    # train_accuracy, test_accuracy = [], []
    # #create logistic regression class
    # reg_lambda, learning_rate, max_iter, eps, batch_size = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 0.00001, 5000, 1e-3, 500
    # for reg in reg_lambda:
    #     print "\nreg_lambda: %f" %(reg)
    #     LG = Logistic_Regression(reg, learning_rate, max_iter, eps, batch_size)
    #     LG.fit(xTrain, yTrain)
    #     train_accuracy.append(LG.best_accuracy)
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
