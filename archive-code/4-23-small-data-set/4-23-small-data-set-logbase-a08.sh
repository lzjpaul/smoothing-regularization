python elasticnet_logistic_regression.py -datapath data/house-votes-84.pkl -onehot 1 -sparsify 0 -batchsize 20 -wlr 3 -maxiter 50000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-5
python elasticnet_logistic_regression.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 50000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-11
python elasticnet_logistic_regression_1.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-23
python elasticnet_logistic_regression_2.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-24