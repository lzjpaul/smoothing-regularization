python elasticnet_logistic_regression_1.py -datapath /data/zhaojing/regularization/uci-dataset/medium_size/Farm_Ads/farm-ads-vect -onehot 0 -sparsify 1 -batchsize 200 -wlr 4 -maxiter 250000 | tee -a /data/zhaojing/regularization/log0425/medium-size/medium-size-3
python elasticnet_logistic_regression_2.py -datapath /data/zhaojing/regularization/uci-dataset/medium_size/Farm_Ads/farm-ads-vect -onehot 0 -sparsify 1 -batchsize 200 -wlr 4 -maxiter 250000 | tee -a /data/zhaojing/regularization/log0425/medium-size/medium-size-4