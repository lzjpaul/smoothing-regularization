import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-datapath', type=str, help='the dataset path, not svm')
    parser.add_argument('-wlr', type=int, help='weight learning_rate (to the power of 10)')
    parser.add_argument('-gmnum', type=int, help='gm_number')
    parser.add_argument('-a', type=float, help='a')
    parser.add_argument('-b', type=float, help='b')
    parser.add_argument('-gmoptmethod', type=int, help='gm optimization method: 0-fixed, 1-GD, 2-EM')
    args = parser.parse_args()
    print "datapath: ", args.datapath
    print "wlr: ", args.wlr
    print "gmnum: ", args.gmnum
    print "a: ", args.a
    print "b: ", args.b
    print "gmoptmethod: ", args.gmoptmethod
