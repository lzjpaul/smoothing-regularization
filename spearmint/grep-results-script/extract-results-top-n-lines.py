# for conn sonar, needs to restrict to first 80 lines
import pandas
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Results')
    parser.add_argument('-resultspath', type=str, help='the results path')
    parser.add_argument('-topn', type=int, help='top n lines')
    args = parser.parse_args()

    print 'resultspath: ', args.resultspath

    resultsfileHandle = open(args.resultspath, "r")
    resultslineList = resultsfileHandle.readlines()
    resultsfileHandle.close()

    print 'len(resultslineList): ', len(resultslineList)

    accuracy_df = pandas.DataFrame()
    accuracy_df.loc[0, (str(resultslineList[0].split()[0]))] = float(resultslineList[1].split()[1])
    accuracy_df.loc[1, (str(resultslineList[0].split()[0]))] = float(resultslineList[2].split()[1])
    accuracy_df.loc[2, (str(resultslineList[0].split()[0]))] = float(resultslineList[3].split()[1])
    accuracy_df.loc[3, (str(resultslineList[0].split()[0]))] = float(resultslineList[4].split()[1])
    accuracy_df.loc[4, (str(resultslineList[0].split()[0]))] = float(resultslineList[5].split()[1])

    for i in range(args.topn):
        if i >= 7:
            if (i-7)%8 == 0:
                accuracy_df.loc[0, (str(resultslineList[i+1].split()[0]))] = float(resultslineList[i+2].split()[1])
                accuracy_df.loc[1, (str(resultslineList[i+1].split()[0]))] = float(resultslineList[i+3].split()[1])
                accuracy_df.loc[2, (str(resultslineList[i+1].split()[0]))] = float(resultslineList[i+4].split()[1])
                accuracy_df.loc[3, (str(resultslineList[i+1].split()[0]))] = float(resultslineList[i+5].split()[1])
                accuracy_df.loc[4, (str(resultslineList[i+1].split()[0]))] = float(resultslineList[i+6].split()[1])

    print accuracy_df

    print("accuracy best subsample %0.6f (+/-%0.06f)"
                         % (accuracy_df.max(axis=1).mean(), accuracy_df.max(axis=1).std()))

# python extract-results-top-n-lines.py -resultspath 19-3-16-spearmint-exp-one-results-7 -topn 100
