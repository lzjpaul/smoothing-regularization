# running folder: /home/singa/zhaojing/smooth-regularization-dbpcm-bak-logistic/smoothing-regularization-tkde-URL-18-9-14
# running machine: logbase gateway
# LR = 1e-7, 1 epoch, only test time
python gm_prior_logistic_regression_sparse_complete_URL_all_param_7.py -datapath /data/zhaojing/regularization/url-dataset/url_svmlight_merge/Day_0_120.svm -onehot 0 -sparsify 1 -batchsize 500 -wlr 7 -pirlr 7 -lambdaslr 7 -maxiter 500000 -gmmuptfreq 100 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log180914/tkde-URL-100-epoch/tkde-URL-100-epoch-11.log
python gm_prior_logistic_regression_sparse_complete_URL_all_param_7.py -datapath /data/zhaojing/regularization/url-dataset/url_svmlight_merge/Day_0_120.svm -onehot 0 -sparsify 1 -batchsize 500 -wlr 7 -pirlr 7 -lambdaslr 7 -maxiter 500000 -gmmuptfreq 100 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log180914/tkde-URL-100-epoch/tkde-URL-100-epoch-11.log