python gm_prior_logistic_regression_tune_small_data_5_fold_even.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-10