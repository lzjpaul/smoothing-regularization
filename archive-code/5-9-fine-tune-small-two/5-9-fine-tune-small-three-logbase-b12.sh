python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-60