# LR = 1e-7, 100 epoch
python gm_prior_logistic_regression_sparse_complete_URL_best.py -datapath /data/zhaojing/regularization/url-dataset/url_svmlight_merge/Day_0_120.svm -onehot 0 -sparsify 1 -batchsize 500 -wlr 7 -pirlr 7 -lambdaslr 7 -maxiter 500000 -gmmuptfreq 100 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0419/URL-100-epoch/URL-100-epoch-6.log
