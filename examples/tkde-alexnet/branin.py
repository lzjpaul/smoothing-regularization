import numpy as np
import math
import datetime
import time
import sys
import os

def branin(lambdaone, lambdatwo, lambdathree, lambdafour):
    print ('brain.py brain() lambdaone: ', lambdaone)
    print ('brain.py brain() lambdatwo: ', lambdatwo)
    print ('brain.py brain() lambdathree: ', lambdathree)
    print ('brain.py brain() lambdafour: ', lambdafour)
    sys.stderr.write("brain.py brain()\n")
    sys.stderr.write ("brain.py brain() time: %s \n" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    lambdasfile=open("alexnet-lambdas.txt","w")
    lambdasfile.write("%f\n"%lambdaone)
    lambdasfile.write("%f\n"%lambdatwo)
    lambdasfile.write("%f\n"%lambdathree)
    lambdasfile.write("%f\n"%lambdafour)
    lambdasfile.close()

    os.system("CUDA_VISIBLE_DEVICES=0 /home/wangwei/miniconda2/bin/python bo_train_no_data_augment.py alexnet cifar-10-batches-py/")

    fileHandle = open("alexnet-result.txt", "r")
    lineList = fileHandle.readlines()
    fileHandle.close()

    result = float(lineList[len(lineList)-1])
    sys.stderr.write ('Result = %f\n' % result)
    print ('print Result = %f\n' % result)
    #time.sleep(np.random.randint(60))
    return result

# Write a function like this called 'main'
def main(job_id, params):
    sys.stderr.write("brain.py main()\n")
    sys.stderr.write('Anything printed here will end up in the output directory for job #%d\n' % job_id)
    sys.stderr.write('params\n')
    print ("check file  in brain.py main() params: ", params)
    return branin(params['lambdaone'][0], params['lambdatwo'][0], params['lambdathree'][0], params['lambdafour'][0])
