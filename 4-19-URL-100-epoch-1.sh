# LR = 1e-7, 100 epoch
python logistic_regression_complete_URL_no_regu.py -datapath /data/zhaojing/regularization/url-dataset/url_svmlight_merge/Day_0_120.svm -onehot 0 -sparsify 1 -batchsize 500 -wlr 7 -maxiter 500000 | tee -a /data/zhaojing/regularization/log0419/URL-100-epoch/URL-100-epoch-1.log