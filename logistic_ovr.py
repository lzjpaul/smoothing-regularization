#my own designed OneVSRestClassifier
#since return from a sigmoid value here, I should change the threshold
import array
import numpy as np
import warnings
import scipy.sparse as sp
from sklearn.multiclass import OneVsRestClassifier,_predict_binary
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.base import MetaEstimatorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import deprecated
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed


class LogisticOneVsRestClassifier(OneVsRestClassifier):
    def predict(self, X):
        """Predict multi-class targets using underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        Returns
        -------
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes].
            Predicted multi-class targets.
        """
        print "in logistic_ovr multiclass.py 181818"
        print "in logistic_ovr multiclass.py 181818 X norm: ", np.linalg.norm(X)
        print "in logistic_ovr multiclass.py 181818 X shape: ", X.shape
        check_is_fitted(self, 'estimators_')
        if (hasattr(self.estimators_[0], "decision_function") and
                is_classifier(self.estimators_[0])):
            thresh = .5
            # since return logistic value, so I should set the threshold to .5 also
        else:
            thresh = .5

        n_samples = _num_samples(X)
        if self.label_binarizer_.y_type_ == "multiclass":
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            for i, e in enumerate(self.estimators_):
                pred = _predict_binary(e, X)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = i
            return self.classes_[np.array(argmaxima.T)]
        else:
            indices = array.array('i')
            indptr = array.array('i', [0])
            for e in self.estimators_:
                indices.extend(np.where(_predict_binary(e, X) > thresh)[0])
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix((data, indices, indptr),
                                      shape=(n_samples, len(self.estimators_)))
            return self.label_binarizer_.inverse_transform(indicator)
