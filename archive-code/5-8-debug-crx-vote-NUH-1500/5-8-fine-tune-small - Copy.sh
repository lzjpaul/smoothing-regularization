# LR not decreasing + converge
python lasso_logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-1
python logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-2
python elasticnet_logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-3

python huber_one_weight_logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-4

python gm_prior_logistic_regression_NUH_1500_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-5

python lasso_logistic_regression_5_fold.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-6
python logistic_regression_5_fold.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-7
python elasticnet_logistic_regression_5_fold.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-8

python huber_one_weight_logistic_regression_5_fold.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-9

python gm_prior_logistic_regression_crx_5_fold.py -datapath data/crx_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-10

python lasso_logistic_regression_5_fold.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-11
python logistic_regression_5_fold.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-12
python elasticnet_logistic_regression_5_fold.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-13

python huber_one_weight_logistic_regression_5_fold.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-14

python gm_prior_logistic_regression_vote_5_fold_1.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-15

python gm_prior_logistic_regression_vote_5_fold_2.py -datapath data/house-votes-84-normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 4 -pirlr 4 -lambdaslr 4 -maxiter 120000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-16