python lasso_logistic_regression.py -datapath data/house-votes-84.pkl -onehot 1 -sparsify 0 -batchsize 20 -wlr 3 -maxiter 50000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-3
python lasso_logistic_regression.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 50000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-9
python lasso_logistic_regression.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-21
python lasso_logistic_regression.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-39
python lasso_logistic_regression.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-57
python lasso_logistic_regression.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -maxiter 500000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-75

python logistic_regression.py -datapath data/house-votes-84.pkl -onehot 1 -sparsify 0 -batchsize 20 -wlr 3 -maxiter 50000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-4
python logistic_regression.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 50000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-10
python logistic_regression.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-22
python logistic_regression.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-40
python logistic_regression.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-58
python logistic_regression.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -maxiter 500000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-76


python elasticnet_logistic_regression.py -datapath data/house-votes-84.pkl -onehot 1 -sparsify 0 -batchsize 20 -wlr 3 -maxiter 50000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-5
python elasticnet_logistic_regression.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 50000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-11
python elasticnet_logistic_regression_1.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-23
python elasticnet_logistic_regression_2.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-24

python elasticnet_logistic_regression_1.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-41
python elasticnet_logistic_regression_2.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-42
python elasticnet_logistic_regression_1.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-59

python elasticnet_logistic_regression_2.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-60
python elasticnet_logistic_regression_1.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -maxiter 500000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-77
python elasticnet_logistic_regression_2.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -maxiter 500000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-78



python huber_one_weight_logistic_regression.py -datapath data/house-votes-84.pkl -onehot 1 -sparsify 0 -batchsize 20 -wlr 3 -maxiter 50000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-6
python huber_one_weight_logistic_regression.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 50000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-12
python huber_one_weight_logistic_regression_1.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-27
python huber_one_weight_logistic_regression_2.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-28
python huber_one_weight_logistic_regression_3.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-29

python huber_one_weight_logistic_regression_4.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-30
python huber_one_weight_logistic_regression_1.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-45
python huber_one_weight_logistic_regression_2.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-46

python huber_one_weight_logistic_regression_3.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-47
python huber_one_weight_logistic_regression_4.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-48
python huber_one_weight_logistic_regression_1.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-63

python huber_one_weight_logistic_regression_2.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-64
python huber_one_weight_logistic_regression_3.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-65
python huber_one_weight_logistic_regression_4.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -maxiter 140000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-66

python huber_one_weight_logistic_regression_1.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -maxiter 500000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-81
python huber_one_weight_logistic_regression_2.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -maxiter 500000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-82

python huber_one_weight_logistic_regression_3.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -maxiter 500000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-83
python huber_one_weight_logistic_regression_4.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -maxiter 500000 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-84


python gm_prior_logistic_regression.py -datapath data/house-votes-84.pkl -onehot 1 -sparsify 0 -batchsize 20 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 50000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-1
python gm_prior_logistic_regression.py -datapath data/house-votes-84.pkl -onehot 1 -sparsify 0 -batchsize 20 -wlr 3 -pirlr 3 -lambdaslr 3 -maxiter 50000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-2
python gm_prior_logistic_regression.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 50000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-7
python gm_prior_logistic_regression.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 50000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-8
python gm_prior_logistic_regression_1.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-13
python gm_prior_logistic_regression_2.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-14
python gm_prior_logistic_regression_3.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-15

python gm_prior_logistic_regression_4.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-16
python gm_prior_logistic_regression_1.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-17
python gm_prior_logistic_regression_2.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-18

python gm_prior_logistic_regression_3.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-19
python gm_prior_logistic_regression_4.py -datapath data/NUH-DIAG.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-20
python gm_prior_logistic_regression_1.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-31

python gm_prior_logistic_regression_2.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-32
python gm_prior_logistic_regression_3.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-33
python gm_prior_logistic_regression_4.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-34

python gm_prior_logistic_regression_1.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-35
python gm_prior_logistic_regression_2.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-36
python gm_prior_logistic_regression_3.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-37

python gm_prior_logistic_regression_4.py -datapath data/NUH-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-38
python gm_prior_logistic_regression_1.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-49
python gm_prior_logistic_regression_2.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-50

python gm_prior_logistic_regression_3.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-51
python gm_prior_logistic_regression_4.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-52
python gm_prior_logistic_regression_1.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-53

python gm_prior_logistic_regression_2.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-54
python gm_prior_logistic_regression_3.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-55
python gm_prior_logistic_regression_4.py -datapath data/NUH-LAB-DIAG-DEMOR.pkl -onehot 0 -sparsify 0 -batchsize 100 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 140000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-56

python gm_prior_logistic_regression_1.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 500000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-67
python gm_prior_logistic_regression_2.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 500000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-68

python gm_prior_logistic_regression_3.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 500000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-69
python gm_prior_logistic_regression_4.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 500000 -gmnum 4 -gmoptmethod 1 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-70

python gm_prior_logistic_regression_1.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 500000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-71
python gm_prior_logistic_regression_2.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 500000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-72

python gm_prior_logistic_regression_3.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 500000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-73
python gm_prior_logistic_regression_4.py -datapath data/diabetic_data_diag_low_dim_2_class_categorical.pkl -onehot 1 -sparsify 0 -batchsize 500 -wlr 6 -pirlr 6 -lambdaslr 6 -maxiter 500000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0423/small-data-set/small-data-set-74

##congression


##NUH-1500

##NUH-DIAG





##NUH-DIAG-DEMOR





##NUH-LAB-DIAG-DEMOR




##uci-diabetes




