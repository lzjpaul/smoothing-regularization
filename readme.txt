Usage:
dbpcm is the main server that modifies the main code ... (not including batchsize=30)

(1) 11-24:
1-1 vec = copy(w)
1-2 add param and C for tuning ratio between params and values

$smoothing_regularization_batch.py: vec= copy(w) && adding param and C altogether && using all-samples
$Example-iris-smoothing-logicbug.py: 
 adding regularization constraint equals 0, such as:
 noreguridge = RidgeClassifier(solver='lsqr')
 param_noreguridge = {'alpha': [0]}

(2) 11-26:
2-1 SGD (use batch of samples), adding permutaion after each epoch
2-2 no vec, but use grad instead

$smoothing_regularization_grad_opti_separate.py: passing the whole X to smoothing_grad_descent, and shuffle here

(3) 11-26:
3-1 loading url dataset

$ Example-url-smoothing.py: using svmlightDataLoader.py to load data

(4) 11-29:
4-1 passing batch to smoothing_grad_descent, and shuffle in smoothing_optimizator!!!
4-2 adding time() to calculate durating

$smoothing_regularization.py: passing batch to smoothing_grad_descent, and shuffle in smoothing_optimizator
$Example-iris-smoothing.py: adding time() to calculate durating 
$Example-url-smoothing.py: adding time() to calculate durating


small changes:
11-29 batchsize = 1 for url-dataset
