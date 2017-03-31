'''
Luo Zhaojing - 2017.3
Implementation of the Dataset
'''

import numpy as np
# import matplotlib.pyplot as plt
import pickle
import argparse

class Dataset(object):
    def DataGenerator(self, x, y):
        self.sample_num, self.dimension = x.shape[0], x.shape[1]
        self.x = x
        self.label = y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-datapath', type=str, help='the dataset path, not svm')
    parser.add_argument('-labelpath', type=str, help='(optional, others are must) the label path, used in NUH data set, not svm')
    parser.add_argument('-outputpath', type=str, help='the output path')
    parser.add_argument('-labelcolumn', type=int, help='labelcolumn, not svm')

    args = parser.parse_args()

    labelcol = args.labelcolumn
    fileName=args.datapath
    labelfile=args.labelpath
    print "labelfile: ", labelfile
    labelCol=(-1 * args.labelcolumn)

    dataset = Dataset()
    if labelfile is None:
        #print "label file is none"
        data = np.loadtxt(fileName, dtype='int32', delimiter=',')
        X, Y = data[:, xrange(data.shape[1]-1) if labelCol==-1 else xrange(1, data.shape[1])], data[:, labelCol]
        print "X shape in loader: ", X.shape
        #print type(OneHotEncoder().fit_transform(X))
    else:
        #print "label file is not none"
        data = np.loadtxt(fileName, dtype='int32', delimiter=',')
        label = np.loadtxt(labelfile, dtype='int32', delimiter=',')
        print "data shape in loader: ", data.shape
        X = data
        Y = label
    dataset.DataGenerator(X, Y)

    # save the generated simulator
    with open(args.outputpath, 'w') as saveFile:
        pickle.dump(dataset, saveFile)

# python pickle_transformer.py -datapath /data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_demor_inpa_kb_ordered_output_onehot_lastcase.csv -labelpath /data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_label.csv -outputpath LACE-CNN-1500-lastcase.pkl -labelcolumn 1 -svmlight 0 -onehot 0 -sparsify 0
# python pickle_transformer.py -datapath /data/zhaojing/regularization/uci-dataset/uci-diabetes-readmission/diag-dim-reduction/diabetic_data_diag_low_dim_diag_3_class_categorical.csv -outputpath diabetic_data_diag_low_dim_diag_3_class_categorical.pkl -labelcolumn 1
# python pickle_transformer.py -datapath /data/zhaojing/regularization/uci-dataset/uci-diabetes-readmission/diag-dim-reduction/diabetic_data_diag_low_dim_3_class_categorical.csv -outputpath diabetic_data_diag_low_dim_3_class_categorical.pkl -labelcolumn 1 -svmlight 0 -sparsify 0
'''
generated results see generated_y_vals.dta
simulator file saved in simulator.pkl

using pkl data:
import pickle
from gm_prior_simulation import Simulator
pickle.load(open('simulator.pkl', 'r'))

probability distribution:	[ 0.2383  0.0896  0.0594  0.0546  0.049   0.0509  0.0536  0.0651  0.0872
  0.2523]
noise misclassification rate:	0.057400
'''
