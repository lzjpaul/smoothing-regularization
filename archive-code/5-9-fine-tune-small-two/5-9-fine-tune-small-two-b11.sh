python gm_prior_logistic_regression_small_data_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-59