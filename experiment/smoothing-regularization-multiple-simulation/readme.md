smoothing-regularization-4-gaussians: (5000, 5000), overfitting
smoothing-regularization-4-gaussians-10000-1000: (10000, 1000), but eraly stopping

(0) gm_prior_logistic_regression_fixed.py !!!!!!!!!!!! --> bias changed (standard)
(1) gm_prior_logistic_regression_fixed_modify --> bias not changed yet, bias still has regularization

3-8
(2) adding loss calculation
(3) mini-batch is sequential fetching samples now (index, index+batchsize)

3-10
(4) adding learning rate decay
(5) train loss for converge
(6) only train and test split

3-13
(7) expectaion-maximization version

3-14
(8) in logistic_regression.py: xTrain = np.hstack((xTrain, np.ones(shape=(self.trainNum, 1)))) the bias is on the lastttt!!!!!
(9) in gm_prior_logistic_regression.py EM version: responsibility * np.square(self.w[:-1]

3-17:
(10) in logistic_regression.py self.w initialization np.random.seed(10)
     the initialization of weights should be the same for different algorithms

3-17:
different from github

