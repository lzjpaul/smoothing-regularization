python gm_prior_logistic_regression_origin_hyper_scale.py -datapath data/bands_normed.pkl -onehot 0 -sparsify 0 -batchsize 30 -wlr 5 -pirlr 5 -lambdaslr 5 -maxiter 360000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log0908/9-8-fine-tune-small-six/9-8-fine-tune-small-six-34.log