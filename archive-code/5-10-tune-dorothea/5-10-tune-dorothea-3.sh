python gm_prior_logistic_regression_sparse_debug_dorothea_a_b_vec_all_param_3.py -datapath /data/zhaojing/regularization/uci-dataset/medium_size/Dorothea/dorothea_train_valid.svm -onehot 0 -sparsify 1 -batchsize 50 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 200000 -gmmuptfreq 50 -gmnum 2 -gmoptmethod 2 | tee -a 5-10-debug-dorothea-3
python gm_prior_logistic_regression_sparse_debug_dorothea_a_b_vec_all_param_3.py -datapath /data/zhaojing/regularization/uci-dataset/medium_size/Dorothea/dorothea_train_valid.svm -onehot 0 -sparsify 1 -batchsize 50 -wlr 5 -pirlr 5 -lambdaslr 5 -maxiter 200000 -gmmuptfreq 50 -gmnum 2 -gmoptmethod 2 | tee -a 5-10-debug-dorothea-3