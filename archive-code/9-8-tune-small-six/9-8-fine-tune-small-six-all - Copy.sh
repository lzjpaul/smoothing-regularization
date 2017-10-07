# small data set
python gm_prior_logistic_regression.py -h
python gm_prior_logistic_regression_origin_hyper_scale.py -h
python gm_prior_logistic_regression_2_time_lambda.py -h
python gm_prior_logistic_regression_2_time_lambda_origin_hyper_scale.py -h

# simulation data set
python gm_prior_logistic_regression.py -h
python gm_prior_logistic_regression_origin_hyper_scale.py -h
python gm_prior_logistic_regression_2_time_lambda.py -h
python gm_prior_logistic_regression_2_time_lambda_origin_hyper_scale.py -h

python huber_one_weight_logistic_regression_1.py -h


# Resnet base-10 : folder: GM-prior-cifar10-base-10
python gm_prior_train.py -h
python gm_prior_train_origin_hyper_scale.py -h

# Alexnet & Resnet
python gm_prior_train_no_data_augment_origin_hyper_scale.py -h
python gm_prior_train_no_data_augment.py -h
python gm_prior_train_origin_hyper_scale.py -h
python gm_prior_train.py -h

# baseline
python train_no_data_augment.py -h
python train.py -h