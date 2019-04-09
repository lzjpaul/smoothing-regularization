# running folder: /home/singa/zhaojing/smooth-regularization-dbpcm-bak-logistic/simulation-data/smoothing-regularization-tkde-true-prob-18-9-9
# running machine: logbase
python huber_one_weight_logistic_regression_3.py -datapath simulator.pkl -onehot 0 -sparsify 0 -batchsize 500 -wlr 5 -maxiter 300000 | tee -a /data/zhaojing/regularization/log180909/tkde-simulator-prob-loss/tkde-simulator-prob-loss-18-09-09-16.log
python huber_one_weight_logistic_regression_4.py -datapath simulator.pkl -onehot 0 -sparsify 0 -batchsize 500 -wlr 5 -maxiter 300000 | tee -a /data/zhaojing/regularization/log180909/tkde-simulator-prob-loss/tkde-simulator-prob-loss-18-09-09-17.log
python huber_one_weight_logistic_regression_3.py -datapath simulator.pkl -onehot 0 -sparsify 0 -batchsize 500 -wlr 5 -maxiter 300000 | tee -a /data/zhaojing/regularization/log180909/tkde-simulator-prob-loss/tkde-simulator-prob-loss-18-09-09-16.log
python huber_one_weight_logistic_regression_4.py -datapath simulator.pkl -onehot 0 -sparsify 0 -batchsize 500 -wlr 5 -maxiter 300000 | tee -a /data/zhaojing/regularization/log180909/tkde-simulator-prob-loss/tkde-simulator-prob-loss-18-09-09-17.log