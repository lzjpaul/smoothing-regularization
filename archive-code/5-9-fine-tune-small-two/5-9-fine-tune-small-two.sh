python lasso_logistic_regression_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-1
python logistic_regression_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-2
python elasticnet_logistic_regression_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-3
python huber_one_weight_logistic_regression_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-4
python gm_prior_logistic_regression_small_data_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-5

python lasso_logistic_regression_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-7
python logistic_regression_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-8
python elasticnet_logistic_regression_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-9
python huber_one_weight_logistic_regression_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-10
python gm_prior_logistic_regression_small_data_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-11


python lasso_logistic_regression_5_fold.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-13
python logistic_regression_5_fold.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-14
python elasticnet_logistic_regression_5_fold.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-15
python huber_one_weight_logistic_regression_5_fold.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-16
python gm_prior_logistic_regression_small_data_5_fold.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-17


python lasso_logistic_regression_5_fold.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-19
python logistic_regression_5_fold.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-20
python elasticnet_logistic_regression_5_fold.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-21
python huber_one_weight_logistic_regression_5_fold.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-22
python gm_prior_logistic_regression_small_data_5_fold.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-23


python lasso_logistic_regression_5_fold.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-31
python logistic_regression_5_fold.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-32
python elasticnet_logistic_regression_5_fold.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-33
python huber_one_weight_logistic_regression_5_fold.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-34
python gm_prior_logistic_regression_small_data_5_fold.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-35



python lasso_logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-43
python logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-44
python elasticnet_logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-45
python huber_one_weight_logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-46
python gm_prior_logistic_regression_small_data_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 5 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-47


python lasso_logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-49
python logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-50
python elasticnet_logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-51
python huber_one_weight_logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-52
python gm_prior_logistic_regression_small_data_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-53


python lasso_logistic_regression_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-55
python logistic_regression_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-56
python elasticnet_logistic_regression_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-57
python huber_one_weight_logistic_regression_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-58
python gm_prior_logistic_regression_small_data_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-59


python lasso_logistic_regression_5_fold.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-61
python logistic_regression_5_fold.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-62
python elasticnet_logistic_regression_5_fold.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-63
python huber_one_weight_logistic_regression_5_fold.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-64
python gm_prior_logistic_regression_small_data_5_fold.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-65


python lasso_logistic_regression_5_fold.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-67
python logistic_regression_5_fold.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-68
python elasticnet_logistic_regression_5_fold.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-69
python huber_one_weight_logistic_regression_5_fold.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-70
python gm_prior_logistic_regression_small_data_5_fold.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-two/fine-tune-small-two-71