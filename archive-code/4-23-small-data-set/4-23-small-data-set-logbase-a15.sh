python huber_one_weight_logistic_regression_4.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-30
python huber_one_weight_logistic_regression_1.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-45
python huber_one_weight_logistic_regression_2.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-46