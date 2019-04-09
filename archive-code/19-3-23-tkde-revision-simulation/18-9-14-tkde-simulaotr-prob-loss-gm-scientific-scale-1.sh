# running folder: /home/singa/zhaojing/smooth-regularization-dbpcm-bak-logistic/simulation-data/smoothing-regularization-tkde-true-prob-18-9-9
# running machine:
# logbase
python gm_prior_logistic_regression_1.py -datapath simulator.pkl -onehot 0 -sparsify 0 -batchsize 500 -wlr 5 -pirlr 5 -lambdaslr 5 -maxiter 300000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log180909/tkde-simulator-prob-loss/tkde-simulator-prob-loss-18-09-09-scientific-scale-1.log
python gm_prior_logistic_regression_2.py -datapath simulator.pkl -onehot 0 -sparsify 0 -batchsize 500 -wlr 5 -pirlr 5 -lambdaslr 5 -maxiter 300000 -gmnum 4 -gmoptmethod 2 | tee -a /data/zhaojing/regularization/log180909/tkde-simulator-prob-loss/tkde-simulator-prob-loss-18-09-09-scientific-scale-2.log