python gm_prior_logistic_regression_3.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 500000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-73
python gm_prior_logistic_regression_4.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 500000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-74