(1) /home/singa/zhaojing/smooth-regularization-dbpcm-bak-logistic/simulation-data/smoothing-regularization-debug-lasso-ridge-etc

(2) /home/singa/zhaojing/smooth-regularization-dbpcm-bak-logistic/singa/incubator-singa/examples/smoothing-regularization-debug-lasso-ridge-etc
python Example-smoothing-regularization-main-all-param-3-25-NUH-1500.py -datapath /data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_demor_inpa_kb_ordered_output_onehot_lastcase.csv -labelpath /data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_label.csv -clf huber -labelcolumn 1 -batchsize 30 -svmlight 0 -sparsify 0 -scale 0 -batchgibbs 0 | tee -a 3-29-debug-huber


some tips:
w_init = variance:0.1
np.random.permutation(10), should rethink random function
huber: w*param instead of 2*param*w
