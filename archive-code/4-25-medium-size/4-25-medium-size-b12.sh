python lasso_logistic_regression.py -datapath /data/zhaojing/regularization/uci-dataset/medium_size/Dorothea/dorothea_train_valid.svm -onehot 0 -sparsify 1 -batchsize 50 -wlr 5 -maxiter 250000 | tee -a /data/zhaojing/regularization/log0425/medium-size/medium-size-49
python logistic_regression.py -datapath /data/zhaojing/regularization/uci-dataset/medium_size/Dorothea/dorothea_train_valid.svm -onehot 0 -sparsify 1 -batchsize 50 -wlr 5 -maxiter 250000 | tee -a /data/zhaojing/regularization/log0425/medium-size/medium-size-50
python huber_one_weight_logistic_regression_1.py -datapath /data/zhaojing/regularization/uci-dataset/medium_size/Dorothea/dorothea_train_valid.svm -onehot 0 -sparsify 1 -batchsize 50 -wlr 5 -maxiter 250000 | tee -a /data/zhaojing/regularization/log0425/medium-size/medium-size-53