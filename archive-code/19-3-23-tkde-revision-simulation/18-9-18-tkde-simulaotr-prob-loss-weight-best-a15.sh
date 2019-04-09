# running folder: /home/singa/zhaojing/smooth-regularization-dbpcm-bak-logistic/simulation-data/smoothing-regularization-tkde-true-prob-18-9-9
# running machine:
# logbase -a14-a16
python logistic_regression_simulation_best.py -datapath simulator.pkl -onehot 0 -sparsify 0 -batchsize 500 -wlr 5 -maxiter 300000 | tee -a /data/zhaojing/regularization/log180918/tkde-simulator-prob-loss-simulation-best/tkde-simulator-prob-loss-simulation-best-18-09-18-2.log
python elasticnet_logistic_regression_simulation_best.py -datapath simulator.pkl -onehot 0 -sparsify 0 -batchsize 500 -wlr 5 -maxiter 300000 | tee -a /data/zhaojing/regularization/log180918/tkde-simulator-prob-loss-simulation-best/tkde-simulator-prob-loss-simulation-best-18-09-18-3.log