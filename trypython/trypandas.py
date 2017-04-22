import pandas
import numpy as np

if __name__ == '__main__':
    np.random.seed(10)
    result_df = pandas.DataFrame()
    for i in range(5):
        reg_mu = [1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1000.]
        reg_lambda = [1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1000.]
        # reg_mu = [1e-4, 1e-3]
        # reg_lambda = [1e-4, 1e-3, 1e-2]
        for mu_val in reg_mu:
            for lambda_val in reg_lambda:
                result_df.loc[i, (str(mu_val) + ',' + str(lambda_val))] = np.random.uniform(0, 1)
    print result_df
    # result_df.loc['Mean'] = result_df.mean()
    # result_df.loc['Std'] = result_df.std()
    pandas.options.display.float_format = '{:,.6f}'.format
    print result_df.values
    print "\n\npandas results\n\n"
    print "mean: ", result_df.mean().values
    print "std: ", result_df.std().values
    print "max mean index: ", result_df.mean().idxmax()
    print "max mean: ", result_df.mean().max()
    print "max mean std: ", result_df.std().loc[result_df.mean().idxmax()]
    print("%0.6f (+/-%0.06f)"
                          % (result_df.mean().max(), result_df.std().loc[result_df.mean().idxmax()]))
    # print "max mean std: ", result_df.loc['Std'].loc[result_df['Mean'].idxmax()]
    print "best each subsample: \n", result_df.max(axis=1)
    print "best each subsample index: \n", result_df.idxmax(axis=1)
    print "mean of each subsample best: ", result_df.max(axis=1).mean()
    print "std of each subsample best: ", result_df.max(axis=1).std()
    print("%0.6f (+/-%0.06f)"
                          % (result_df.max(axis=1).mean(), result_df.max(axis=1).std()))
