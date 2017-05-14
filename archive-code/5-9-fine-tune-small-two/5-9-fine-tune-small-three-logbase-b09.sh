python lasso_logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-49
python logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-50
python elasticnet_logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-51
python huber_one_weight_logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-52
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-53