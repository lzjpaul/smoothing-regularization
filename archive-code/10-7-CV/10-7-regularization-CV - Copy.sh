#5-8:  LR not decreasing + converge + LR is 1e-2 scale, bigger L2 (vote 84 is smaller LR: 1e-3 scale)
#5-8 yes
python lasso_logistic_regression_CV.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-1.log
python logistic_regression_CV.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-2.log
python elasticnet_logistic_regression_CV.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-3.log
python huber_one_weight_logistic_regression_CV.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-4.log

python gm_prior_logistic_regression_CV.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-5.log

#5-8 yes
python lasso_logistic_regression_CV.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-6.log
python logistic_regression_CV.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-7.log
python elasticnet_logistic_regression_CV.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-8.log
python huber_one_weight_logistic_regression_CV.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-9.log

python gm_prior_logistic_regression_CV.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-10.log

#5-8 yes
python lasso_logistic_regression_CV.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-11.log
python logistic_regression_CV.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-12.log
python elasticnet_logistic_regression_CV.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-13.log
python huber_one_weight_logistic_regression_CV.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-14.log

python gm_prior_logistic_regression_CV.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-15.log

#5-9:  LR is 1e-3 scale, smaller L2
#5-9 yes
python lasso_logistic_regression_CV.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-16.log
python logistic_regression_CV.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-17.log
python elasticnet_logistic_regression_CV.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-18.log
python huber_one_weight_logistic_regression_CV.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-19.log

python gm_prior_logistic_regression_CV.py -datapath data/breast-cancer-wisconsin-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-20.log

#5-9 yes
python lasso_logistic_regression_CV.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-21.log
python logistic_regression_CV.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-22.log
python elasticnet_logistic_regression_CV.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-23.log
python huber_one_weight_logistic_regression_CV.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-24.log

python gm_prior_logistic_regression_CV.py -datapath data/wdbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-25.log

#5-9 yes
python lasso_logistic_regression_CV.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-26.log
python logistic_regression_CV.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-27.log
python elasticnet_logistic_regression_CV.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-28.log
python huber_one_weight_logistic_regression_CV.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-29.log

python gm_prior_logistic_regression_CV.py -datapath data/wpbc_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-30.log

#5-9 yes
python lasso_logistic_regression_CV.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-31.log
python logistic_regression_CV.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-32.log
python elasticnet_logistic_regression_CV.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-33.log
python huber_one_weight_logistic_regression_CV.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-34.log

python gm_prior_logistic_regression_CV.py -datapath data/pop_failures_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-35.log

#5-9 yes
python lasso_logistic_regression_CV.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-36.log
python logistic_regression_CV.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-37.log
python elasticnet_logistic_regression_CV.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-38.log
python huber_one_weight_logistic_regression_CV.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-39.log

python gm_prior_logistic_regression_CV.py -datapath data/sonar-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-40.log

#5-10 iter = 36000
python lasso_logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-41.log
python logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-42.log
python elasticnet_logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-43.log
python huber_one_weight_logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 360000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-44.log

python gm_prior_logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 5 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-45.log

#5-9 try
python lasso_logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-46.log
python logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-47.log
python elasticnet_logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-48.log
python huber_one_weight_logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-49.log

python gm_prior_logistic_regression_CV.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 5 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-50.log

#5-9 yes
python lasso_logistic_regression_CV.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-51.log
python logistic_regression_CV.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-52.log
python elasticnet_logistic_regression_CV.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-53.log
python huber_one_weight_logistic_regression_CV.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-54.log

python gm_prior_logistic_regression_CV.py -datapath data/hepatitis_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-55.log

#5-9 yes
python lasso_logistic_regression_CV.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-56.log
python logistic_regression_CV.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-57.log
python elasticnet_logistic_regression_CV.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-58.log
python huber_one_weight_logistic_regression_CV.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-59.log

python gm_prior_logistic_regression_CV.py -datapath data/horse-colic-train-test-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-60.log

#5-9 yes
python lasso_logistic_regression_CV.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-61.log
python logistic_regression_CV.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-62.log
python elasticnet_logistic_regression_CV.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-63.log
python huber_one_weight_logistic_regression_CV.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-64.log

python gm_prior_logistic_regression_CV.py -datapath data/ionosphere_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log1007/regularization-CV/regularization-CV-65.log