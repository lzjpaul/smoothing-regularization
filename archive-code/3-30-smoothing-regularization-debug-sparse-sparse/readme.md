be careful of the hyper, for example. reg_lambda is the same as logistic_regression.py
first version: w and y are sparse
second version: w and y are dense

 python gm_prior_logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 1 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 3 -maxiter 900 -gmnum 3 -a 1 -b 10 -alpha 1 -gmoptmethod 1
  1022  python gm_prior_logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 1 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 3 -maxiter 900 -gmnum 3 -a 1 -b 10 -alpha 1 -gmoptmethod 2
   1023  python gm_prior_logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 1 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 3 -maxiter 900 -gmnum 3 -a 1 -b 10 -alpha 1 -gmoptmethod 1
    1024  python gm_prior_logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 1 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 3 -maxiter 900 -gmnum 3 -a 1 -b 10 -alpha 1 -gmoptmethod 0
     1025  python huber_logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 1 -batchsize 30 -wlr 5 -maxiter 5000
      1026  python elasticnet_logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 1 -batchsize 30 -wlr 5 -maxiter 5000
       1027  python logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 1 -batchsize 30 -wlr 5 -maxiter 5000
        1028  python lasso_logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 1 -batchsize 30 -wlr 5 -maxiter 5000
