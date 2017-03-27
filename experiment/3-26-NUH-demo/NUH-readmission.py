import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse
from onehotencoder import onehottransformer
class Logistic_Regression(object):
    def read_w(self, filename):
        self.w = np.genfromtxt(filename, delimiter=',').reshape((-1,1))

    # predict probability
    def predict_proba(self, samples):
        if samples.shape[1] != self.w.shape[0]:
            samples = np.hstack((samples, np.ones(shape=(samples.shape[0], 1))))
        return self.sigmoid(np.matmul(samples, self.w))

    # predict probability, no normalization
    def predict_proba_no_normal(self, samples):
        if samples.shape[1] != self.w.shape[0]:
            samples = np.hstack((samples, np.ones(shape=(samples.shape[0], 1))))
        return np.matmul(samples, self.w)


    def auroc(self, yPredictProba, yTrue):
        return roc_auc_score(yTrue, yPredictProba)

    # predict result
    def predict(self, samples):
        if samples.shape[1] != self.w.shape[0]:
            samples = np.hstack((samples, np.ones(shape=(samples.shape[0], 1))))
        return np.matmul(samples, self.w)>0.0

    # sigmoid function
    def sigmoid(self, matrix):
        return 1.0/(1.0+np.exp(-matrix))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-wpath', type=str, help='the weight path')
    parser.add_argument('-datapath', type=str, help='the dataset path')
    args = parser.parse_args()

    LG = Logistic_Regression()
    print '\n===============================================\n'
    print 'loading model...'
    LG.read_w(args.wpath)
    print 'loading data...'
    samples = np.genfromtxt(args.datapath, delimiter=',')
    print 'sample number %d'%(samples.shape[0])
    print 'feature number %d'%(samples.shape[1])
    print '\n===============================================\n'

    # onehotencoder
    samples = onehottransformer('data-param/nuh_fa_readmission_case_demor_inpa_kb_ordered_output_categorical.csv', args.datapath)
    print "samples shape after onehot: ", samples.shape

    print 'readmission probability: \n'
    print LG.predict_proba(samples).reshape(-1,1)
    print '\n'
    # print LG.predict_proba_no_normal(samples).reshape(-1,1)
    all_samples = np.genfromtxt('data-param/nuh_fa_readmission_case_demor_inpa_kb_ordered_output_onehot_lastcase.csv', delimiter=',')
    print 'AUC: ', LG.auroc(LG.predict_proba_no_normal(all_samples)[1455:1755],(np.genfromtxt('data-param/nuh_fa_readmission_case_label.csv', \
            delimiter=',')>0.5)[1455:1755])
