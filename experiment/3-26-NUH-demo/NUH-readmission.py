import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse

class Logistic_Regression(object):
    def read_w(self, filename):
        self.w = np.genfromtxt(filename, delimiter=',').reshape((-1,1))

    # predict probability
    def predict_proba(self, samples):
        if samples.shape[1] != self.w.shape[0]:
            samples = np.hstack((samples, np.ones(shape=(samples.shape[0], 1))))
        return self.sigmoid(np.matmul(samples, self.w))

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

    print 'readmission probability: \n'
    print LG.predict_proba(samples).reshape(-1,1)
    # print 'AUC: ', LG.auroc(LG.predict_proba(samples)[:30],(np.genfromtxt('/data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_label.csv', \
    #        delimiter=',')>0.5)[:30])
