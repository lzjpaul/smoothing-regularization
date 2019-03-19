import numpy as np
import math
import datetime
import time
import sys
import os

def branin(x):
    print ('brain.py brain() x: ', x)
    sys.stderr.write("brain.py brain()\n")
    sys.stderr.write ("brain.py brain() time: %s \n" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    os.system("CUDA_VISIBLE_DEVICES=0 /home/wangwei/miniconda2/bin/python train.py resnet cifar-10-batches-py/")
    fileHandle = open("resnet-result.txt", "r")
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
    print ("in brain.py main() params: ", params)
    return branin(params['x'][0])
