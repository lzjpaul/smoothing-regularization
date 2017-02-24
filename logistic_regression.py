'''
Cai Shaofeng - 2017.2
Implementation of the Logistic Regression
'''

from data_loader import *

# base logistic regression class
class Logistic_Regression():
    def __init__(self, reg_lambda=1, learning_rate=0.1, max_iter=1000, eps=1e-4, batch_size=-1):
        self.reg_lambda, self.learning_rate = reg_lambda, learning_rate
        self.max_iter, self.eps, self.batch_size = max_iter, eps, batch_size

    # calc the delta w to update w, using newton's method here
    def delta_w(self, xTrain, yTrain):
        # mini batch, not used here
        if self.batch_size != -1:
            randomIndex = np.random.random_integers(0, self.trainNum-1, self.batch_size)
            xTrain, yTrain = xTrain[randomIndex], yTrain[randomIndex]

        mu = self.sigmoid(np.matmul(xTrain, self.w))
        # check here, no regularization over bias term
        grad_w = np.matmul(xTrain.T, (mu-yTrain)) + np.vstack(([0], np.full((self.w.shape[0]-1,1), self.reg_lambda, dtype='float32')))*self.w
        #grad_w = np.matmul(xTrain.T, (mu - yTrain)) + self.reg_lambda * self.w
        S = np.diag((mu*(1-mu)).reshape((xTrain.shape[0])))
        hessian = np.matmul(xTrain.T, np.matmul(S, xTrain)) + self.reg_lambda*np.identity(self.w.shape[0])
        return -np.matmul(np.linalg.pinv(hessian), grad_w)

    def fit(self, xTrain, yTrain, xTest, yTest):
        # find the number of class and feature, allocate memory for model parameters
        self.trainNum, self.featureNum = xTrain.shape[0], xTrain.shape[1]
        self.w = np.zeros(shape=(self.featureNum+1, 1), dtype='float32')#np.random.normal(0, 1, size=(self.featureNum+1, 1))

        # adding 1s to each training examples
        xTrain = np.hstack((np.ones(shape=(self.trainNum, 1)), xTrain))

        try:
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

                test_accuracy = self.accuracy(self.predict(xTest), yTest)
                train_accuracy = self.accuracy(self.predict(xTrain), yTrain)
                if self.best_accuracy < test_accuracy:
                    self.best_accuracy, self.best_iter = test_accuracy, iter
                print "iter %4d\t|\ttrain_accuracy %10.6f\t|\ttest_accuracy %10.6f\t|\tbest_accuracy %10.6f"\
                      %(iter, train_accuracy, test_accuracy, self.best_accuracy)
        except:
            pass
        finally:
            print self


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
        return 'model parameter {\treg: %.6f, lr: %.6f, batch_size: %5d, best_iter: %6d, best_accuracy: %.6f\t}' \
            % (self.reg_lambda, self.learning_rate, self.batch_size, self.best_iter, self.best_accuracy)

if __name__ == '__main__':
    # load the simulation data
    xTrain, xTest, yTrain, yTest = loadData('simulator.pkl', trainPerc=0.7)

    # create logistic regression class
    reg_lambda, learning_rate, max_iter, eps, batch_size = 1, 0.1, 100, 1e-3, 7000
    LG = Logistic_Regression(reg_lambda, learning_rate, max_iter, eps, batch_size)
    LG.fit(xTrain, yTrain, xTest, yTest)


'''
model parameter {	reg: 1.000000, lr: 0.100000, batch_size:  7000, best_iter:     39, best_accuracy: 0.900667	}
'''

