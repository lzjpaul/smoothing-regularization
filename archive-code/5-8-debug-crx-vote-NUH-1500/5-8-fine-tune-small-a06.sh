python lasso_logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-1
python logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-2
python elasticnet_logistic_regression_5_fold.py -datapath data/LACE-CNN-1500-lastcase.pkl -onehot 0 -sparsify 0 -batchsize 50 -wlr 4 -maxiter 120000 | tee -a /data/zhaojing/regularization/log0508/fine-tune-small/fine-tune-small-3