# LR is 1e-3 scale, smaller lambda + iter == 36 0000
python lasso_logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-1
python logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-2
python elasticnet_logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-3
python huber_one_weight_logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-4
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 5 -pirlr 5 -lambdaslr 5 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-5

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 5 -pirlr 5 -lambdaslr 5 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-6

python lasso_logistic_regression_5_fold.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-7
python logistic_regression_5_fold.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-8
python elasticnet_logistic_regression_5_fold.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-9
python huber_one_weight_logistic_regression_5_fold.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-10
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 5 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-11

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 5 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-12

python lasso_logistic_regression_5_fold.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-13
python logistic_regression_5_fold.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-14
python elasticnet_logistic_regression_5_fold.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-15
python huber_one_weight_logistic_regression_5_fold.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-16
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-17

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-18

python lasso_logistic_regression_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-19
python logistic_regression_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-20
python elasticnet_logistic_regression_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-21
python huber_one_weight_logistic_regression_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-22
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-23

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-24

python lasso_logistic_regression_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-25
python logistic_regression_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-26
python elasticnet_logistic_regression_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-27
python huber_one_weight_logistic_regression_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-28
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-29

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-30


python lasso_logistic_regression_5_fold.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-31
python logistic_regression_5_fold.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-32
python elasticnet_logistic_regression_5_fold.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-33
python huber_one_weight_logistic_regression_5_fold.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-34
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-35

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-36


python lasso_logistic_regression_5_fold.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-37
python logistic_regression_5_fold.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-38
python elasticnet_logistic_regression_5_fold.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-39
python huber_one_weight_logistic_regression_5_fold.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-40
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-41

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-42


python lasso_logistic_regression_5_fold.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-43
python logistic_regression_5_fold.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-44
python elasticnet_logistic_regression_5_fold.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-45
python huber_one_weight_logistic_regression_5_fold.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-46
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-47

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-48


python lasso_logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-49
python logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-50
python elasticnet_logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-51
python huber_one_weight_logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-52
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 5 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-53

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 5 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-54


python lasso_logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-55
python logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-56
python elasticnet_logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-57
python huber_one_weight_logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-58
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-59

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-60


python lasso_logistic_regression_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-61
python logistic_regression_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-62
python elasticnet_logistic_regression_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-63
python huber_one_weight_logistic_regression_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-64
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-65

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-66


python lasso_logistic_regression_5_fold.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-67
python logistic_regression_5_fold.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-68
python elasticnet_logistic_regression_5_fold.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-69
python huber_one_weight_logistic_regression_5_fold.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-70
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-71

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-72


python lasso_logistic_regression_5_fold.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-73
python logistic_regression_5_fold.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-74
python elasticnet_logistic_regression_5_fold.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-75
python huber_one_weight_logistic_regression_5_fold.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-76
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-77

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0510/fine-tune-small-four/fine-tune-small-four-78