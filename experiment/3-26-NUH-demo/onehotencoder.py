import numpy as np
import numpy
from sklearn.preprocessing import OneHotEncoder
import sys

#not include ID column


def onehottransformer(onehotfilename, transformfilename):
    #define model
    train_X = np.genfromtxt(onehotfilename, delimiter=',')[:, 1:].astype(np.int)
    print "train_X[1000]: ", train_X[1000]
    print "train_X shape = \n", train_X.shape

    test_X = np.genfromtxt(transformfilename, delimiter=',')[:, 0:].astype(np.int)
    print "test_X shape = \n", test_X.shape

    #OneHotEncoder
    num_feature = len(train_X[0,:])
    enc = OneHotEncoder(categorical_features=range(0,num_feature),sparse=False)  #modify here!!
    print "categorical_features = \n", range(0,num_feature)
    print "num_feature: ", num_feature
    enc.fit(train_X)
    test_X = enc.transform(test_X)
    test_X = test_X.astype(int)
    print "test_X shape: ", test_X.shape
    print "test_X[0,1]: "
    print test_X[0:2]
    return test_X
#python hotencoder_arg.py NUH_DS_SOC_READMISSION_CASE_halfyear_LAB_ENGI_SUB_idxcase_demor.txt NUH_DS_SOC_READMISSION_CASE_halfyear_LAB_ENGI_SUB_idxcase_demor_onehot_arg.txt
