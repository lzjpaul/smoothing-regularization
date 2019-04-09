import numpy as np
import math
import datetime
import time
import sys
import os

def branin(lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19, lambda_20, lambda_21, lambda_22):
    print ('brain.py brain() lambda_1: ', lambda_1)
    print ('brain.py brain() lambda_2: ', lambda_2)
    print ('brain.py brain() lambda_3: ', lambda_3)
    print ('brain.py brain() lambda_4: ', lambda_4)
    print ('brain.py brain() lambda_5: ', lambda_5)
    print ('brain.py brain() lambda_6: ', lambda_6)
    print ('brain.py brain() lambda_7: ', lambda_7)
    print ('brain.py brain() lambda_8: ', lambda_8)
    print ('brain.py brain() lambda_9: ', lambda_9)
    print ('brain.py brain() lambda_10: ', lambda_10)
    print ('brain.py brain() lambda_11: ', lambda_11)
    print ('brain.py brain() lambda_12: ', lambda_12)
    print ('brain.py brain() lambda_13: ', lambda_13)
    print ('brain.py brain() lambda_14: ', lambda_14)
    print ('brain.py brain() lambda_15: ', lambda_15)
    print ('brain.py brain() lambda_16: ', lambda_16)
    print ('brain.py brain() lambda_17: ', lambda_17)
    print ('brain.py brain() lambda_18: ', lambda_18)
    print ('brain.py brain() lambda_19: ', lambda_19)
    print ('brain.py brain() lambda_20: ', lambda_20)
    print ('brain.py brain() lambda_21: ', lambda_21)
    print ('brain.py brain() lambda_22: ', lambda_22)
    sys.stderr.write("brain.py brain()\n")
    sys.stderr.write ("brain.py brain() time: %s \n" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    lambdasfile=open("resnet-lambdas.txt","w")
    lambdasfile.write("%f\n"%lambda_1)
    lambdasfile.write("%f\n"%lambda_2)
    lambdasfile.write("%f\n"%lambda_3)
    lambdasfile.write("%f\n"%lambda_4)
    lambdasfile.write("%f\n"%lambda_5)
    lambdasfile.write("%f\n"%lambda_6)
    lambdasfile.write("%f\n"%lambda_7)
    lambdasfile.write("%f\n"%lambda_8)
    lambdasfile.write("%f\n"%lambda_9)
    lambdasfile.write("%f\n"%lambda_10)
    lambdasfile.write("%f\n"%lambda_11)
    lambdasfile.write("%f\n"%lambda_12)
    lambdasfile.write("%f\n"%lambda_13)
    lambdasfile.write("%f\n"%lambda_14)
    lambdasfile.write("%f\n"%lambda_15)
    lambdasfile.write("%f\n"%lambda_16)
    lambdasfile.write("%f\n"%lambda_17)
    lambdasfile.write("%f\n"%lambda_18)
    lambdasfile.write("%f\n"%lambda_19)
    lambdasfile.write("%f\n"%lambda_20)
    lambdasfile.write("%f\n"%lambda_21)
    lambdasfile.write("%f\n"%lambda_22)
    lambdasfile.close()

    os.system("CUDA_VISIBLE_DEVICES=0 /home/wangwei/miniconda2/bin/python bo_train.py resnet cifar-10-batches-py/")

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
    print ("check file in brain.py main() params: ", params)
    return branin(params['lambda_1'][0], params['lambda_2'][0], params['lambda_3'][0], params['lambda_4'][0], params['lambda_5'][0], params['lambda_6'][0], params['lambda_7'][0], params['lambda_8'][0], params['lambda_9'][0], params['lambda_10'][0], params['lambda_11'][0], params['lambda_12'][0], params['lambda_13'][0], params['lambda_14'][0], params['lambda_15'][0], params['lambda_16'][0], params['lambda_17'][0], params['lambda_18'][0], params['lambda_19'][0], params['lambda_20'][0], params['lambda_21'][0], params['lambda_22'][0])
