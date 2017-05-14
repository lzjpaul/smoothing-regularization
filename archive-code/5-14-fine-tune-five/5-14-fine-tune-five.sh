#breast cancer, wdbc, glass, hepatitis (a, b, 0.2, 0.4, 0.6, 0.8, alpha 0.2, 0.4, 0.6, 0.8) + uci-diabetest (5 fold)
python gm_prior_logistic_regression_tune_small_data_5_fold.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-1

python gm_prior_logistic_regression_tune_small_data_5_fold_even.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-2

python gm_prior_logistic_regression_tune_small_data_5_fold_even_alpha.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-3

python gm_prior_logistic_regression_tune_small_data_5_fold_even_even_alpha.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-4

python gm_prior_logistic_regression_tune_small_data_5_fold.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-5

python gm_prior_logistic_regression_tune_small_data_5_fold_even.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-6

python gm_prior_logistic_regression_tune_small_data_5_fold_even_alpha.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-7

python gm_prior_logistic_regression_tune_small_data_5_fold_even_even_alpha.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-8

python gm_prior_logistic_regression_tune_small_data_5_fold.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-9

python gm_prior_logistic_regression_tune_small_data_5_fold_even.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-10

python gm_prior_logistic_regression_tune_small_data_5_fold_even_alpha.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-11

python gm_prior_logistic_regression_tune_small_data_5_fold_even_even_alpha.py -datapath data/glass_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-12

python gm_prior_logistic_regression_tune_small_data_5_fold.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-13

python gm_prior_logistic_regression_tune_small_data_5_fold_even.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-14

python gm_prior_logistic_regression_tune_small_data_5_fold_even_alpha.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-15

python gm_prior_logistic_regression_tune_small_data_5_fold_even_even_alpha.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-16


python gm_prior_logistic_regression_tune_uci_diabetes_5_fold_1.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-17

python gm_prior_logistic_regression_tune_uci_diabetes_5_fold_2.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-18

python gm_prior_logistic_regression_tune_uci_diabetes_5_fold_3.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-19

python gm_prior_logistic_regression_tune_uci_diabetes_5_fold_4.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-20

python lasso_logistic_regression_5_fold.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-21

python logistic_regression_5_fold.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-22

python elasticnet_logistic_regression_5_fold.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-23

python huber_one_weight_logistic_regression_5_fold.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0514/fine-tune-five/fine-tune-five-24