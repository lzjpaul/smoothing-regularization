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

3-20
(11) merge GM EM GD into one programme

3-23
(12) adding w_loss (prior w)
(13) the initialization is arbitrary
(14) a, b, alpha are parameters

3-25
(15) enrich input type (pickle_transformer.py)

3-27
(16) adding ealsticnet, huber, Lasso (data_grad is separated out)

3-28
(17) sparse version + data_loader permutation + huber no w1_lr + self.likelihood_grad

3-29
(18) check lasso, elasticnet, huber. etc
18-1: import logistic_regression class
18-2: the likelihood grad is different here (1.0/batch_size) VS (train_num/batch_size)
18-3: np.random.norm, np.random.permutation
18-4: huber L2: w*param instead of 2*w*param (old version of author's code is not correct)

3-30
(19) sparse version (first version slow, second version fast)
(20) sparse and dense combined version (in GD: responsibility*np.square(self.w)!!!) + y is NOT returned as sparse
smoothing-regularization (dense, all checked correct, now archive-code/3-30-dense) --> smoothing-regularization-debug-sparse-dense(authority) &&  smoothing-regularization-debug-sparse-sparse(authority)(all checked, correct)
--> smoothing-regularization (dense + sparse combined, all checked correct)
(21) seed is followed by np.random.permutation(normal) + data_loader permutation once --> stratifiedKfold

4-1
(22) all-param.py

4-3
(23) special sparsity technique (gm_prior_logistic_regression_sparse.py)
     --- if sparsify
4-4
(24) sparsify + all param (gm_prior_logistic_regression_sparse_all_param.py)
(25) all the algorithms turn to be all_param.py, copy from archive-code/all-param
script:

04-02-all-param.sh: a and b choose according to [1e-4, 0.1, 0.5, 1., 2.], [0.5, 1.]
because this is the best for simulator.pkl(a, b not tuned for NUH-1500 and uci-diabetes!!)
4-4-URL.sh: URL data set, a and b choose according to [1e-4, 0.1, 0.5, 1., 2.], [0.5, 1.], not tuned!!

4-5
(26) delete validation set

4-9
(27) print test-loss
4-8-test-loss.sh: the same as 04-02-all-param.sh and 4-4-URL.sh, but this time print test loss (URL data set using the best params suggested by SF/JY because only have 8 machines)

##############up till now, a and b are only tuned for simulator.pkl#############################

4-10
0410-healthcare-a-b.sh: tune a, b for NUH-1500, uci-diabetes
and
0410-simulator-gmm-lr.sh: tunelambda_s_lr and pi_r_lr for simulator.pkl (choose the a, b according to best test loss)

4-11
!!!URL: for sparse implementation, update pi and lambda (iter < 100 or iter % 100 == 0)
pirlr, lambdaslr: only 1e-8 works
4-8-test-loss-new-URL.sh: copy 4-8-test-loss.sh but lambda_s lr and pi_r lr is 1e-8 only
4-11-test-loss-EM.sh: copy 0410-healthcare-a-b.sh and 4-8-test-loss-new-URL.sh, but the gmoptmethod is 2 (EM)
