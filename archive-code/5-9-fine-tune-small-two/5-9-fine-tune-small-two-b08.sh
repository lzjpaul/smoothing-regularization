python lasso_logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-49
python logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-50
python elasticnet_logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-51
python huber_one_weight_logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-52