import numpy as np
from sklearn.preprocessing import OneHotEncoder

def classificationDataLoader(fileName, labelfile,  labelCol=-1, delimiter=',', sparsify=True):
    '''load classification data from fileName, transform into onehot feature representation X, and label Y
        :param
            fileName:   data file
            labelCol:   column position for label, default last column
            delimiter:  feature seperator each sample, default ','
        :returns
            X:  np feature matrix   (n_samples, n_features)
            Y:  np label matrix     (n_samples, 1)
    '''
    if labelfile is None:
        #print "label file is none"
        data = np.loadtxt(fileName, dtype='int32', delimiter=',')
        X, Y = data[:, xrange(data.shape[1]-1) if labelCol==-1 else xrange(1, data.shape[1])], data[:, labelCol]
        #print type(OneHotEncoder().fit_transform(X))
    else:
        #print "label file is not none"
        data = np.loadtxt(fileName, dtype='int32', delimiter=',')
        label = np.loadtxt(labelfile, dtype='int32', delimiter=',')
        print "data shape in loader: ", data.shape
        X = data
        Y = label
    return (OneHotEncoder().fit_transform(X, ), Y) if sparsify==True else (OneHotEncoder().fit_transform(X, ).toarray(), Y)

# X, Y = classificationDataLoader('dataset/test.data')
# print X.shape, Y.shape
# print X, Y
