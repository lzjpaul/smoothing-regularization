 1010  python gm_prior_logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 0 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 3 -maxiter 900 -gmnum 3 -a 1 -b 10 -alpha 1 -gmoptmethod 1
  1011  python huber_logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 5000
   1012  python elasticnet_logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 5000
    1013  python lasso_logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 5000
     1014  python logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 5000
      1015  python gm_prior_logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 0 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 3 -maxiter 900 -gmnum 3 -a 1 -b 10 -alpha 1 -gmoptmethod 0
       1016  python gm_prior_logistic_regression.py -datapath LACE-CNN-1500-lastcase.pkl -onehot 1 -sparsify 0 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 3 -maxiter 900 -gmnum 3 -a 1 -b 10 -alpha 1 -gmoptmethod 2
