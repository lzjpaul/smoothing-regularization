# lr is set to 1e-8 and total iterations is 12500 only
python gm_prior_logistic_regression_sparse_URL_best_1_3.py -datapath /data/zhaojing/regularization/url-dataset/url_svmlight_merge/Day_0_120.svm -onehot 0 -sparsify 1 -batchsize 500 -wlr 5 -pirlr 8 -lambdaslr 8 -maxiter 12500 -gmmuptfreq 100 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0408/test-loss/test-loss-new-3