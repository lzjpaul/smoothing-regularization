'''
Luo Zhaojing - 2017.2
Implementation of the Simulator
'''
'''
need to modify self.w_origin = np.random.choice(2, size=(self.dimension), p=self.pi)
2 is number of gaussian numbers
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print Weight')
    parser.add_argument('-weightpath', default='cifar-10-batches-py')
    parser.add_argument('-savepath', default='save.png')
    args = parser.parse_args()

    weight = np.genfromtxt(args.weightpath, delimiter=',')
    print 'weight shape: '
    print weight.shape
    plt.hist(weight, bins=50, normed=1, color='g', alpha=0.75)
    plt.savefig(args.savepath)
    plt.show()

'''
python plot_weight_save_png.py -weightpath weight-out/simulation-best-result/logistic_regression_simulation_best_w.out -savepath weight-out/save_images/logistic_save.png
weight shape: 
(1001,)

'''

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
'''
2018-9-9
optimal accuracy:  0.13244
probability distribution:distribution[ 0.31718  0.06432  0.04568  0.03804  0.03414  0.03574  0.0378   0.04558
  0.06638  0.31514]
  noise misclassification rate:rate0.069360
'''
'''
2018-9-18
optimal accuracy:  0.11762
probability distribution:	[ 0.33878  0.05878  0.03902  0.03314  0.03118  0.03126  0.03206  0.04028
  0.05578  0.33972]
noise misclassification rate:	0.060680
'''


