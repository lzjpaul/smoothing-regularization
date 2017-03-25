[singa@logbase smoothing-regularization-multiple-simulation]$ vim gm_prior_multiple_simulations.py
[singa@logbase smoothing-regularization-multiple-simulation]$ python gm_prior_multiple_simulations.py 2
simulation pi, variance, sum(pi):  [ 0.85  0.15] [ 0.005  1.   ] 1.0
optimal accuracy:  0.0542
probability distribution:	[ 0.4399  0.0296  0.0165  0.0171  0.0121  0.0134  0.0132  0.0169  0.0252
  0.4143  0.0018]
noise misclassification rate:	0.026400
simulation pi, variance, sum(pi):  [ 0.85  0.15] [ 1.     0.005] 1.0
optimal accuracy:  0.0232
probability distribution:	[ 0.4682  0.0102  0.008   0.0055  0.0054  0.0056  0.0058  0.008   0.0101
  0.3578  0.1154]
noise misclassification rate:	0.011200
simulation pi, variance, sum(pi):  [ 0.5  0.5] [ 0.005  1.   ] 1.0
optimal accuracy:  0.0281
probability distribution:	[ 0.4702  0.0139  0.0081  0.0082  0.0074  0.0066  0.0071  0.0094  0.0146
  0.4034  0.0511]
noise misclassification rate:	0.015800
[singa@logbase smoothing-regularization-multiple-simulation]$ python gm_prior_multiple_simulations.py 4
simulation pi, variance, sum(pi):  [ 0.7   0.05  0.2   0.05] [ 0.005  0.05   0.1    0.8  ] 1.0
optimal accuracy:  0.0839
probability distribution:	[ 0.3872  0.0417  0.0282  0.0236  0.0198  0.0225  0.0236  0.028   0.0381
  0.3873]
noise misclassification rate:	0.046400
simulation pi, variance, sum(pi):  [ 0.7   0.05  0.2   0.05] [ 0.8    0.1    0.05   0.005] 1.0
optimal accuracy:  0.0278
probability distribution:	[ 0.4716  0.0132  0.0094  0.0079  0.005   0.0066  0.0076  0.0087  0.0142
  0.3923  0.0635]
noise misclassification rate:	0.013200
simulation pi, variance, sum(pi):  [ 0.7   0.05  0.2   0.05] [ 0.003  0.005  0.8    1.   ] 1.0
optimal accuracy:  0.0505
probability distribution:	[ 0.4324  0.0236  0.0146  0.0154  0.0109  0.0124  0.0121  0.0157  0.0227
  0.4371  0.0031]
noise misclassification rate:	0.023100
simulation pi, variance, sum(pi):  [ 0.7   0.05  0.2   0.05] [ 1.     0.8    0.005  0.003] 1.0
optimal accuracy:  0.0245
probability distribution:	[ 0.4647  0.0093  0.0072  0.0068  0.0069  0.0065  0.006   0.0083  0.0106
  0.3812  0.0925]
noise misclassification rate:	0.012700
simulation pi, variance, sum(pi):  [ 0.5   0.4   0.05  0.05] [ 0.005  0.05   0.1    0.8  ] 1.0
optimal accuracy:  0.0786
probability distribution:	[ 0.4008  0.0387  0.0233  0.02    0.0178  0.0181  0.0208  0.0244  0.0413
  0.3948]
noise misclassification rate:	0.043300
simulation pi, variance, sum(pi):  [ 0.5   0.4   0.05  0.05] [ 0.8    0.1    0.05   0.005] 1.0
optimal accuracy:  0.0305
probability distribution:	[ 0.4626  0.0145  0.0096  0.0083  0.0083  0.0081  0.0066  0.01    0.0133
  0.4257  0.033 ]
noise misclassification rate:	0.015900
simulation pi, variance, sum(pi):  [ 0.5   0.4   0.05  0.05] [ 0.003  0.005  0.8    1.   ] 1.0
optimal accuracy:  0.0795
probability distribution:	[ 0.4041  0.0385  0.0248  0.0192  0.0206  0.0187  0.0214  0.0213  0.0367
  0.3947]
noise misclassification rate:	0.043100
simulation pi, variance, sum(pi):  [ 0.5   0.4   0.05  0.05] [ 1.     0.8    0.005  0.003] 1.0
optimal accuracy:  0.0228
probability distribution:	[ 0.477   0.0101  0.0071  0.0052  0.0064  0.005   0.0067  0.0085  0.0109
  0.3616  0.1015]
noise misclassification rate:	0.011200
simulation pi, variance, sum(pi):  [ 0.25  0.25  0.25  0.25] [ 0.005  0.05   0.1    0.8  ] 1.0
optimal accuracy:  0.0418
probability distribution:	[ 0.438   0.0195  0.0141  0.0116  0.01    0.0111  0.0111  0.0139  0.0207
  0.441   0.009 ]
noise misclassification rate:	0.022600
simulation pi, variance, sum(pi):  [ 0.25  0.25  0.25  0.25] [ 0.003  0.005  0.8    1.   ] 1.0
optimal accuracy:  0.0303
probability distribution:	[ 0.4571  0.0175  0.0106  0.0082  0.0084  0.0077  0.0078  0.0119  0.0137
  0.4106  0.0465]
noise misclassification rate:	0.014400
[singa@logbase smoothing-regularization-multiple-simulation]$ ls
data                                   gm_prior_logistic_regression.py   simulator2-1.pkl  simulator4-6.pkl
data_loader.py                         gm_prior_multiple_simulations.py  simulator2-2.pkl  simulator4-7.pkl
data_loader.pyc                        gm_prior_simulation.py            simulator4-0.pkl  simulator4-8.pkl
early-stop-version                     gm_prior_simulation.pyc           simulator4-1.pkl  simulator4-9.pkl
experiment                             logistic_regression.py            simulator4-2.pkl  tmp
generated_y_vals.dta                   logistic_regression.pyc           simulator4-3.pkl  trypython
gm_prior_logistic_regression_em.py     readme.md                         simulator4-4.pkl  weight-out
gm_prior_logistic_regression_fixed.py  simulator2-0.pkl                  simulator4-5.pkl
[singa@logbase smoothing-regularization-multiple-simulation]$ python gm_prior_multiple_simulations.py 8
simulation pi, variance, sum(pi):  [ 0.2    0.3    0.2    0.2    0.025  0.025  0.025  0.025] [ 0.002  0.005  0.03   0.05   0.1    0.3    0.8    1.   ] 1.0
optimal accuracy:  0.0802
probability distribution:	[ 0.3873  0.0359  0.0245  0.0213  0.0186  0.0212  0.021   0.0271  0.0406
  0.4025]
noise misclassification rate:	0.038700
simulation pi, variance, sum(pi):  [ 0.2    0.3    0.2    0.2    0.025  0.025  0.025  0.025] [ 1.     0.8    0.3    0.1    0.05   0.03   0.005  0.002] 1.0
optimal accuracy:  0.0281
probability distribution:	[ 0.466   0.0129  0.0093  0.0075  0.0063  0.0057  0.0072  0.0085  0.0143
  0.4028  0.0595]
noise misclassification rate:	0.015900
simulation pi, variance, sum(pi):  [ 0.2    0.3    0.2    0.2    0.025  0.025  0.025  0.025] [ 0.001   0.0015  0.002   0.0025  0.5     0.7     0.9     1.    ] 1.0
optimal accuracy:  0.0796
probability distribution:	[ 0.3981  0.0377  0.0278  0.0202  0.0187  0.0189  0.0209  0.0252  0.0374
  0.3951]
noise misclassification rate:	0.039800
simulation pi, variance, sum(pi):  [ 0.2    0.3    0.2    0.2    0.025  0.025  0.025  0.025] [ 1.      0.9     0.7     0.5     0.0025  0.002   0.0015  0.001 ] 1.0
optimal accuracy:  0.0268
probability distribution:	[ 0.4672  0.0117  0.0095  0.0075  0.0057  0.0065  0.0076  0.0081  0.0139
  0.3797  0.0826]
noise misclassification rate:	0.013800
simulation pi, variance, sum(pi):  [ 0.35   0.35   0.1    0.1    0.025  0.025  0.025  0.025] [ 0.002  0.005  0.03   0.05   0.1    0.3    0.8    1.   ] 1.0
optimal accuracy:  0.0715
probability distribution:	[ 0.4015  0.0351  0.0237  0.0202  0.0196  0.0203  0.021   0.0249  0.0349
  0.3988]
noise misclassification rate:	0.037800
simulation pi, variance, sum(pi):  [ 0.35   0.35   0.1    0.1    0.025  0.025  0.025  0.025] [ 1.     0.8    0.3    0.1    0.05   0.03   0.005  0.002] 1.0
optimal accuracy:  0.0278
probability distribution:	[ 0.4736  0.0133  0.0085  0.0062  0.0061  0.0057  0.0069  0.0083  0.0121
  0.3796  0.0797]
noise misclassification rate:	0.015100
simulation pi, variance, sum(pi):  [ 0.35   0.35   0.1    0.1    0.025  0.025  0.025  0.025] [ 0.001   0.0015  0.002   0.0025  0.5     0.7     0.9     1.    ] 1.0
optimal accuracy:  0.0756
probability distribution:	[ 0.3963  0.0363  0.0248  0.023   0.0184  0.0181  0.0211  0.0253  0.0361
  0.4006]
noise misclassification rate:	0.040200
simulation pi, variance, sum(pi):  [ 0.35   0.35   0.1    0.1    0.025  0.025  0.025  0.025] [ 1.      0.9     0.7     0.5     0.0025  0.002   0.0015  0.001 ] 1.0
optimal accuracy:  0.0212
probability distribution:	[ 0.4659  0.0111  0.0058  0.0059  0.006   0.0047  0.0056  0.0094  0.0111
  0.3686  0.1059]
noise misclassification rate:	0.011700
simulation pi, variance, sum(pi):  [ 0.125  0.125  0.125  0.125  0.125  0.125  0.125  0.125] [ 0.002  0.005  0.03   0.05   0.1    0.3    0.8    1.   ] 1.0
optimal accuracy:  0.0367
probability distribution:	[ 0.455   0.0208  0.0112  0.0099  0.0101  0.0104  0.0103  0.0107  0.0198
  0.4269  0.0149]
noise misclassification rate:	0.021000
simulation pi, variance, sum(pi):  [ 0.125  0.125  0.125  0.125  0.125  0.125  0.125  0.125] [ 0.001   0.0015  0.002   0.0025  0.5     0.7     0.9     1.    ] 1.0
optimal accuracy:  0.032
probability distribution:	[ 0.4645  0.0173  0.0093  0.0078  0.0092  0.0087  0.0077  0.0092  0.0154
  0.418   0.0329]
noise misclassification rate:	0.016500
