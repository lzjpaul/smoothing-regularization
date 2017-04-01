0. code changed
0-1 7:3 split, not permutation (data loader using the previous one)
0-2 no cross-validation
0-3 a,b,alpha is using list

1. objective:
1-1 find the best a, b, alpha
1-2 find the best simulation pi&alpha combination

2. steps:
(1) simulator.pkl: GMM-fixed-1 -- tune a, b, alpha
(2) simulator4-10.pkl: try L2 different lambda and GMM-fixed-1 to see the difference
pi: [0.70, 0.05, 0.2, 0.05], lambda: [0.005, 0.008, 0.1, 0.8]
(3) simulator.pkl: 5-fold or 3-fold?
