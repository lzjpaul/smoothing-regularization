python gm_prior_logistic_regression_sparse_alpha_3_b_1.py -datapath /data/zhaojing/regularization/uci-dataset/medium_size/Dorothea/dorothea_train_valid.svm -onehot 0 -sparsify 1 -batchsize 50 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 250000 -gmmuptfreq 50 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0427/medium-tune-alpha/medium-tune-alpha-13