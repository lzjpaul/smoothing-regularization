# This file contains all the classes that are related to 
# BO regularizer.
# =============================================================================
from singa import optimizer
from singa.optimizer import SGD
from singa.optimizer import Regularizer
from singa.optimizer import Optimizer
import numpy as np
from singa import tensor
from singa import singa_wrap as singa
from singa.proto import model_pb2
from scipy.stats import norm as gaussian
import math

class BOOptimizer(Optimizer):
    '''
    introduce hyper-parameters for GM-regularization: a, b, alpha
    '''
    def __init__(self, net=None, lr=None, momentum=None, weight_decay=None,
                 regularizer=None, constraint=None):
        Optimizer.__init__(self, lr=lr, momentum=momentum, weight_decay=weight_decay,
                 regularizer=regularizer, constraint=constraint)
        # self.gmregularizer = GMRegularizer(hyperpara=hyperpara, gm_num=gm_num, pi=pi, reg_lambda=reg_lambda)

    def calcRegGrad(self, dev, bo_reg_lambda, net, epoch, value, grad, name, step):
        grad.to_device(dev)
        tensor.axpy(bo_reg_lambda, value, grad)
        return grad

    def apply_BO_regularizer_constraint(self, dev, bo_reg_lambda, net, epoch, value, grad, name, step):
        # if np.ndim(tensor.to_numpy(value)) <= 2:
        if np.ndim(tensor.to_numpy(value)) != 2:
            return self.apply_regularizer_constraint(epoch, value, grad, name, step)
        else: # weight parameter
            grad = self.calcRegGrad(dev, bo_reg_lambda, net, epoch, value, grad, name, step)
            return grad

class BOSGD(BOOptimizer, SGD):
    '''The vallina Stochasitc Gradient Descent algorithm with momentum.
    But this SGD has a BO regularizer
    '''

    def __init__(self, net=None, lr=None, momentum=None, weight_decay=None,
                 regularizer=None, constraint=None):
        BOOptimizer.__init__(self, net=net, lr=lr, momentum=momentum, weight_decay=weight_decay, regularizer=regularizer,
                                  constraint=constraint)
        SGD.__init__(self, lr=lr, momentum=momentum, weight_decay=weight_decay,
                 regularizer=regularizer, constraint=constraint)

    # compared with apply_with_lr, this need one more argument: isweight
    def apply_with_lr(self, dev, bo_reg_lambda, net, epoch, lr, grad, value, name, step=-1):
        if grad.is_empty():
            return value
        ##### using bo_regularizer ##############
        grad = self.apply_BO_regularizer_constraint(dev=dev, bo_reg_lambda=bo_reg_lambda, net=net, epoch=epoch, value=value, grad=grad, name=name, step=step)
        ##### using bo_regularizer ##############
        if name is not None and name in self.learning_rate_multiplier:
            lr = lr * self.learning_rate_multiplier[name]
        self.opt.Apply(epoch, lr, name, grad.singa_tensor, value.singa_tensor)
        return value
