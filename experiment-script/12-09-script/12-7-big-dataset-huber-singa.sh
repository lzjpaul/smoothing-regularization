# /log1209/
python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/uci-dataset/car_evaluation/car.categorical.data -labelcolumn 1 -svmlight 0 -sparsify 0 -scale 1 | tee -a /data/zhaojing/regularization/log1209/carscale1.log
python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/uci-dataset/car_evaluation/car.categorical.data -labelcolumn 1 -svmlight 0 -sparsify 0 -scale 0 | tee -a /data/zhaojing/regularization/log1209/carscale0.log

python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/uci-dataset/congression/house-votes-84.categorical.data -labelcolumn 0 -svmlight 0 -sparsify 0 -scale 1 | tee -a /data/zhaojing/regularization/log1209/congressionscale1.log
python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/uci-dataset/congression/house-votes-84.categorical.data -labelcolumn 0 -svmlight 0 -sparsify 0 -scale 0 | tee -a /data/zhaojing/regularization/log1209/congressionscale0.log

python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/uci-dataset/balance_scale/balance-scale.categorical.data -labelcolumn 0 -svmlight 0 -sparsify 0 -scale 1 | tee -a /data/zhaojing/regularization/log1209/balancescale1.log
python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/uci-dataset/balance_scale/balance-scale.categorical.data -labelcolumn 0 -svmlight 0 -sparsify 0 -scale 0 | tee -a /data/zhaojing/regularization/log1209/balancescale0.log

python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/uci-dataset/Audiology/audio_data/audiology.standardized.traintestcategorical.data -labelcolumn 1 -svmlight 0 -sparsify 0 -scale 1 | tee -a /data/zhaojing/regularization/log1209/audiologyscale1.log
python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/uci-dataset/Audiology/audio_data/audiology.standardized.traintestcategorical.data -labelcolumn 1 -svmlight 0 -sparsify 0 -scale 0 | tee -a /data/zhaojing/regularization/log1209/audiologyscale0.log

python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/uci-dataset/dermatology/dermatology.categorical.csv -labelcolumn 1 -svmlight 0 -sparsify 0 -scale 1 | tee -a /data/zhaojing/regularization/log1209/dermatologyscale1.log
python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/uci-dataset/dermatology/dermatology.categorical.csv -labelcolumn 1 -svmlight 0 -sparsify 0 -scale 0 | tee -a /data/zhaojing/regularization/log1209/dermatologyscale0.log

python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/url-dataset/url_svmlight_merge/Day0.svm -labelcolumn 1 -svmlight 1 -sparsify 1 -scale 0 | tee -a /data/zhaojing/regularization/log1209/Day0cale0.log

python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/uci-dataset/uci-diabetes-readmission/diabetic_data_categorical.csv -labelcolumn 1 -svmlight 0 -sparsify 1 -scale 1 | tee -a /data/zhaojing/regularization/log1209/diabeticscale1.log
python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/uci-dataset/uci-diabetes-readmission/diabetic_data_categorical.csv -labelcolumn 1 -svmlight 0 -sparsify 1 -scale 0 | tee -a /data/zhaojing/regularization/log1209/diabeticscale0.log

python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/VISIT_DIAG_aggcnt_1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_data_1case.csv -labelpath /data/zhaojing/regularization/VISIT_DIAG_aggcnt_1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_label_1case.csv -labelcolumn 1 -svmlight 0 -sparsify 1 -scale 1 | tee -a /data/zhaojing/regularization/log1209/aggcntscale1.log
python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/VISIT_DIAG_aggcnt_1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_data_1case.csv -labelpath /data/zhaojing/regularization/VISIT_DIAG_aggcnt_1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_label_1case.csv -labelcolumn 1 -svmlight 0 -sparsify 1 -scale 0 | tee -a /data/zhaojing/regularization/log1209/aggcntscale0.log

python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/intersect/NUH_DS_SOC_READMISSION_CASE_DIAG_LAB_ENGI_SUB_idxcase_DIAG_INTERSECT.txt -labelpath /data/zhaojing/regularization/intersect/NUH_DS_SOC_READMISSION_CASE_LAB_idxcase_label_INTERSECT.txt -labelcolumn 1 -svmlight 0 -sparsify 1 -scale 1 | tee -a /data/zhaojing/regularization/log1209/DIAGscale1.log
python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/intersect/NUH_DS_SOC_READMISSION_CASE_DIAG_LAB_ENGI_SUB_idxcase_DIAG_INTERSECT.txt -labelpath /data/zhaojing/regularization/intersect/NUH_DS_SOC_READMISSION_CASE_LAB_idxcase_label_INTERSECT.txt -labelcolumn 1 -svmlight 0 -sparsify 1 -scale 0 | tee -a /data/zhaojing/regularization/log1209/DIAGscale0.log

python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/intersect/concat/check-regularization/NUH_DS_SOC_READMISSION_CASE_DIAG_LAB_ENGI_SUB_idxcase_DIAG_demor_onehot_INTERSECT.txt -labelpath /data/zhaojing/regularization/intersect/NUH_DS_SOC_READMISSION_CASE_LAB_idxcase_label_INTERSECT.txt -labelcolumn 1 -svmlight 0 -sparsify 1 -scale 1 | tee -a /data/zhaojing/regularization/log1209/DIAGdemorscale1.log
python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/intersect/concat/check-regularization/NUH_DS_SOC_READMISSION_CASE_DIAG_LAB_ENGI_SUB_idxcase_DIAG_demor_onehot_INTERSECT.txt -labelpath /data/zhaojing/regularization/intersect/NUH_DS_SOC_READMISSION_CASE_LAB_idxcase_label_INTERSECT.txt -labelcolumn 1 -svmlight 0 -sparsify 1 -scale 0 | tee -a /data/zhaojing/regularization/log1209/DIAGdemorscale0.log

python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/intersect/concat/check-regularization/NUH_DS_SOC_READMISSION_CASE_DIAG_LAB_ENGI_SUB_idxcase_first_and_last_DIAG_demor_onehot_INTERSECT.txt -labelpath /data/zhaojing/regularization/intersect/NUH_DS_SOC_READMISSION_CASE_LAB_idxcase_label_INTERSECT.txt -labelcolumn 1 -svmlight 0 -sparsify 1 -scale 1 | tee -a /data/zhaojing/regularization/log1209/labDIAGdemorscale1.log
python /home/singa/zhaojing/smooth-regularization/smoothing-regularization/Example-iris-smoothing.py -datapath /data/zhaojing/regularization/intersect/concat/check-regularization/NUH_DS_SOC_READMISSION_CASE_DIAG_LAB_ENGI_SUB_idxcase_first_and_last_DIAG_demor_onehot_INTERSECT.txt -labelpath /data/zhaojing/regularization/intersect/NUH_DS_SOC_READMISSION_CASE_LAB_idxcase_label_INTERSECT.txt -labelcolumn 1 -svmlight 0 -sparsify 1 -scale 0 | tee -a /data/zhaojing/regularization/log1209/labDIAGdemorscale0.log