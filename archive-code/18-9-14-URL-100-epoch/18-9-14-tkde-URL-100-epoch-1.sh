# running folder: /home/singa/zhaojing/smooth-regularization-dbpcm-bak-logistic/smoothing-regularization-tkde-URL-18-9-14
# running machine: logbase gateway
# LR = 1e-7, 1 epoch, only test time
python logistic_regression_complete_URL_no_regu.py -datapath /data/zhaojing/regularization/url-dataset/url_svmlight_merge/Day_0_120.svm -onehot 0 -sparsify 1 -batchsize 500 -wlr 7 -maxiter 500000 | tee -a /data/zhaojing/regularization/log180914/tkde-URL-100-epoch/tkde-URL-100-epoch-0.log
python logistic_regression_complete_URL_no_regu.py -datapath /data/zhaojing/regularization/url-dataset/url_svmlight_merge/Day_0_120.svm -onehot 0 -sparsify 1 -batchsize 500 -wlr 7 -maxiter 500000 | tee -a /data/zhaojing/regularization/log180914/tkde-URL-100-epoch/tkde-URL-100-epoch-0.log