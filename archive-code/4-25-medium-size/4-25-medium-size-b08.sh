python lasso_logistic_regression.py -datapath /data/zhaojing/regularization/uci-dataset/medium_size/Dorothea/dorothea_train_valid.svm -onehot 0 -sparsify 1 -batchsize 50 -wlr 4 -maxiter 250000 | tee -a /data/zhaojing/regularization/log0425/medium-size/medium-size-41
python logistic_regression.py -datapath /data/zhaojing/regularization/uci-dataset/medium_size/Dorothea/dorothea_train_valid.svm -onehot 0 -sparsify 1 -batchsize 50 -wlr 4 -maxiter 250000 | tee -a /data/zhaojing/regularization/log0425/medium-size/medium-size-42
python huber_one_weight_logistic_regression_1.py -datapath /data/zhaojing/regularization/uci-dataset/medium_size/Dorothea/dorothea_train_valid.svm -onehot 0 -sparsify 1 -batchsize 50 -wlr 4 -maxiter 250000 | tee -a /data/zhaojing/regularization/log0425/medium-size/medium-size-45