# LR is 1e-2 scale, bigger lambda + iter == 36 0000
python lasso_logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-1
python logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-2
python elasticnet_logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-3
python huber_one_weight_logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-4
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-5

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-6

python lasso_logistic_regression_5_fold.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-7
python logistic_regression_5_fold.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-8
python elasticnet_logistic_regression_5_fold.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-9
python huber_one_weight_logistic_regression_5_fold.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-10
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-10

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-12

python lasso_logistic_regression_5_fold.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-13
python logistic_regression_5_fold.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-14
python elasticnet_logistic_regression_5_fold.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-15
python huber_one_weight_logistic_regression_5_fold.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-16
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-17

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-18

python lasso_logistic_regression_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-19
python logistic_regression_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-20
python elasticnet_logistic_regression_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-21
python huber_one_weight_logistic_regression_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-22
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-23

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-24

python lasso_logistic_regression_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-25
python logistic_regression_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-26
python elasticnet_logistic_regression_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-27
python huber_one_weight_logistic_regression_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-28
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-29

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-30


python lasso_logistic_regression_5_fold.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-31
python logistic_regression_5_fold.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-32
python elasticnet_logistic_regression_5_fold.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-33
python huber_one_weight_logistic_regression_5_fold.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-34
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-35

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-36


python lasso_logistic_regression_5_fold.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-37
python logistic_regression_5_fold.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-38
python elasticnet_logistic_regression_5_fold.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-39
python huber_one_weight_logistic_regression_5_fold.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-40
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-41

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-42


python lasso_logistic_regression_5_fold.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-43
python logistic_regression_5_fold.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-44
python elasticnet_logistic_regression_5_fold.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-45
python huber_one_weight_logistic_regression_5_fold.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-46
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-47

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-48


python lasso_logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-49
python logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-50
python elasticnet_logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-51
python huber_one_weight_logistic_regression_5_fold.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-52
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-53

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-54


python lasso_logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-55
python logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-56
python elasticnet_logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-57
python huber_one_weight_logistic_regression_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-58
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-59

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-60


python lasso_logistic_regression_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-61
python logistic_regression_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-62
python elasticnet_logistic_regression_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-63
python huber_one_weight_logistic_regression_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-64
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-65

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-66


python lasso_logistic_regression_5_fold.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-67
python logistic_regression_5_fold.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-68
python elasticnet_logistic_regression_5_fold.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-69
python huber_one_weight_logistic_regression_5_fold.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-70
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-71

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-72


python lasso_logistic_regression_5_fold.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-73
python logistic_regression_5_fold.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-74
python elasticnet_logistic_regression_5_fold.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-75
python huber_one_weight_logistic_regression_5_fold.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -maxiter 360000 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-76
python gm_prior_logistic_regression_tune_small_data_5_fold_1.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-77

python gm_prior_logistic_regression_tune_small_data_5_fold_2.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0509/fine-tune-small-three/fine-tune-small-three-78