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
    def delta_w(self, xTrain, yTrain):
        # mini batch, not used here
        if self.batch_size != -1:
            randomIndex = np.random.random_integers(0, xTrain.shape[0]-1, self.batch_size)
            xTrain, yTrain = xTrain[randomIndex], yTrain[randomIndex]

        mu = self.sigmoid(np.matmul(xTrain, self.w))
        # check here, no regularization over bias term # need normalization with xTrain.shape[0]/batch_size here
        grad_w = (self.trainNum/self.batch_size)*np.matmul(xTrain.T, (mu - yTrain))
        print 'reg', self.reg_lambda
        print 'w', self.w
        print 'reg_w', self.reg_lambda * self.w
        print 'grad_w', grad_w
        grad_w += self.reg_lambda * self.w
        return -grad_w

    def fit(self, xTrain, yTrain):
        # find the number of class and feature, allocate memory for model parameters
        self.trainNum, self.featureNum = xTrain.shape[0], xTrain.shape[1]
        self.w = np.random.normal(0, 0.01, size=(self.featureNum+1, 1))#np.zeros(shape=(self.featureNum+1, 1), dtype='float32')

        # adding 1s to each training examples
        xTrain = np.hstack((np.ones(shape=(self.trainNum, 1)), xTrain))

        # validation set
        validationNum = int(self.validation_perc*xTrain.shape[0])
        xVallidation, yVallidation = xTrain[:validationNum, ], yTrain[:validationNum, ]
        xTrain, yTrain = xTrain[validationNum:, ], yTrain[validationNum:, ]


        # try:
        iter, self.best_accuracy, self.best_iter = 0, 0.0, 0
        while True:
            # calc the delta_w to update w
            delta_w = self.delta_w(xTrain, yTrain)
            # update w
            self.w += self.learning_rate * delta_w

            # stop updating if converge
            iter += 1
            if iter > self.max_iter or np.linalg.norm(delta_w, ord=2) < self.eps:
                break

            if iter % 100 == 0:
                # print np.sum(np.abs(self.w))/self.featureNum, np.linalg.norm(self.w, ord=2)
                test_accuracy, train_accuracy = self.accuracy(self.predict(xVallidation), yVallidation), self.accuracy(self.predict(xTrain), yTrain)
                if self.best_accuracy < test_accuracy:
                     self.best_w, self.best_accuracy, self.best_iter = np.copy(self.w), test_accuracy, iter
                print "iter %4d\t|\ttrain_accuracy %10.6f\t|\ttest_accuracy %10.6f\t|\tbest_accuracy %10.6f"\
                      %(iter, train_accuracy, test_accuracy, self.best_accuracy)
        # except:
        #     pass
        # finally:
        #     self.w = self.best_w

    # predict result
    def predict(self, samples):
        if samples.shape[1] != self.w.shape[0]:
            samples = np.hstack((np.ones(shape=(samples.shape[0], 1)), samples))
        return np.matmul(samples, self.w)>0.5

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

    reg_lambda, learning_rate, max_iter, eps, batch_size = 10, 0.0001, 2000, 1e-3, 500
    print "\nreg_lambda: %f" % (reg_lambda)
    LG = Logistic_Regression(reg_lambda, learning_rate, max_iter, eps, batch_size)
    LG.fit(xTrain, yTrain)
    print "\n\nfinal accuracy: %.6f" % (LG.accuracy(LG.predict(xTest), yTest))
    print LG, LG.best_w


    # train_accuracy, test_accuracy = [], []
    # #create logistic regression class
    # reg_lambda, learning_rate, max_iter, eps, batch_size = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 0.1, 5000, 1e-3, 500
    # for reg in reg_lambda:
    #     print "\nreg_lambda: %f" %(reg)
    #     LG = Logistic_Regression(reg, learning_rate, max_iter, eps, batch_size)
    #     LG.fit(xTrain, yTrain)
    #     train_accuracy.append(LG.best_accuracy)
    #     test_accuracy.append(LG.accuracy(LG.predict(xTest), yTest))
    #     print "finally accuracy: %.6f" %(test_accuracy[-1])
    #     print LG

    # fig, ax = plt.subplots()
    # ax.plot(reg_lambda, train_accuracy, 'r-', label='train', ); ax.plot(reg_lambda, test_accuracy, 'b-', label='test')
    # ax.set_xscale('log'); ax.set_xticks(reg_lambda); plt.xlabel('reg_lambda');plt.ylabel('accuracy');plt.legend(loc='upper right')
    # plt.title('accuracy VS reg_lambda'); plt.savefig('data/l2_accuracy.eps', format='eps', dpi=1000)
    # plt.show()


'''
best reg_lamdba in range [0.1, 10], around 1, most likely to be [1, 10]

'''

