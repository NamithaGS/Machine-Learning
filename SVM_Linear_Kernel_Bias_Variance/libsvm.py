import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from svmutil import *
from scipy import io
import random
from datetime import datetime, date
import SVM_Linear_Kernel_Bias_Variance as r

if __name__ == "__main__":
     datatrain = io.loadmat('phishing-train.mat')
     datatest = io.loadmat('phishing-test.mat')
     dftrainx,dftrainy = r.datapreprocess (datatrain)
     dftestx,dftesty = r.datapreprocess (datatest)
     classes = dftrainy["Y"].values
     data =dftrainx.values.tolist()
     problem = svm_problem(classes, data)

     print " Best Kernel : RBF Kernel"
     cvalues = [1]
     gamma  = [-1]
     print " CValue : ", cvalues[0]
     print " Gamma : ", gamma[0]
     for eachc in cvalues:
         cvaccuracy =0.0
         avgtrainingtime = 0.0
         eachc = pow(4,eachc)
         for eachgamma in gamma:
             eachgamma = pow(4,eachgamma)
             param = svm_parameter('-q')
             param.kernel_type = 1
             param.cost = eachc
             param.gamma = eachgamma
             model = svm_train(problem, param)
             current_time1 = datetime.now().time()
             p_label, p_acc, p_val = svm_predict(dftesty["Y"],dftestx.values.tolist(), model,options="-q")
             current_time2 = datetime.now().time()
             avgtrainingtime = (datetime.combine(date.today(), current_time2) - datetime.combine(date.today(), current_time1))/3

     print " Crossvalidation accuracy on Testset : " ,  p_acc[0]
     print " Average Training time on Testset    : " , avgtrainingtime
