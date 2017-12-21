python lasso_logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-41.log
python logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-42.log
python elasticnet_logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-43.log
python huber_one_weight_logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-44.log