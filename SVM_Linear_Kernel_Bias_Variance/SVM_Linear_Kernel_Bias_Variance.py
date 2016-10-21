import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from svmutil import *
from scipy import io
import random
from datetime import datetime, date
import math

title = ["g1(x) = 1 ","g2(x) = w0","g3(x) = w0 + w1x","g4(x)=w0 + w1x + w2x2","g5(x)=w0+ w1x + w2x2 + w2x3","g6(x)= w0+ w1x + w2x2 + w3x3 + w4x4"]
def  msecalc(ytrue,ypred):
    diff =0.0
    N = len(ytrue)
    for i in range(0,N):
        a = ytrue[i]
        b = ypred[i]
        diff1 = a - b
        diff = diff +  math.pow(diff1,2)
    return diff/float(N)

def calcw (pdtrainx,pdtrainy):
     a1=[]
     pdtrainxt = pdtrainx.T
     a1 = pdtrainxt.dot(pdtrainx)
     if a1.size == 1:
         if (a1[0]==0):
             inverse = 0.0
         else:
            inverse = np.reciprocal(a1)
     else:
         inverse = np.linalg.pinv(a1)
     a2 = np.dot(inverse, pdtrainxt)
     ans = np.dot(a2,pdtrainy)
     return ans


def calcypred(hd,eachrowtestx):
    ypred= np.dot(hd.T,eachrowtestx.T)
    return ypred

def getyforx(x):
     ans = 0.0
     a = 2.0 * (x*x)
     e = (1/(math.sqrt( 2.0 * 3.1415 * 0.1 ) ) ) * ( math.exp( (0.0-(x*x)) / (2.0* 0.1 )))
     ans = a+e
     return ans

def getyforxwithoutnoise(x):
     a = 0.0
     a = 2.0 * x * x
     return a

def calchd(eachrowx,eachlinehd,index):
    ans = 0.0
    if index == 1:
        ans = ans + 1.0
    elif index == 2:
        ans  = ans + eachlinehd[0]
    else :
        i = 0
        for eachhd in eachlinehd:
            ans = float(ans) +(float(eachhd)  * float(math.pow(eachrowx,i)))
            i = i + 1
    return float(ans)

def calceedhd(hd,index,eachrowx):
    ans =0.0
    for eachlinehd in hd:
        ans = float(ans) +  float(calchd(eachrowx,eachlinehd,index))
    hd1 = float(ans)/float(len(hd))
    return hd1

def  calcbiasvariance(datatrainx, datatrainy,datatestx,datatesty , index, dataones, size):
     mse = []
     hd =[]
     for eachrowx,eachrowy in zip(datatrainx,datatrainy):
        thishd =[]
        thishd =calcw(eachrowx, eachrowy)
        hd.append(thishd)
        if index == 1:
            ypredtrain = np.ones((size,1))
        elif index ==2:
            ypredtrain = np.tile(thishd[0][0],(size,1))
        else :
            ypredtrain = calcypred(thishd,eachrowx)[0]
        mse.append(msecalc(eachrowy[1:], ypredtrain))

     #MSE on every dataset and draw
     #mse1 = [mse[i].item() for i in range (0,len(mse))]
     mse1 = mse
     bias2 = 0.0
     variance = 0.0

     for eachrowx,eachrowy in zip(datatestx,datatesty):
         temp = calceedhd(hd,index,eachrowx)
         if( (index!=1) and (index!=2)):
             eachrowy = eachrowx

         aa = float(temp) - float(eachrowy)
         bias2 = float(bias2) + float((aa*aa))
     for eachlinehd in hd:
        for (eachrowx,eachrowy) in zip(datatestx,datatesty):
             bb =  float(calchd(eachrowx,eachlinehd,index)) - float(calceedhd(hd,index,eachrowx))
             variance = float(variance) + (bb*bb)
     actualbias = float(bias2)/float(len(datatestx))
     actualvariance = float(variance)/float((len(datatestx)*len(hd)))
     return mse1,actualbias,actualvariance

def plothist(actualmse, samplesize):
    index=0
    for eachmse in actualmse:
         plt.subplot(2,3,index+1)
         plt.title(title[index])
         plt.ylabel("Frequency")
         plt.xlabel("MSE")
         plt.hist(eachmse,bins=10)

         index = index+ 1
    #plt.tight_layout()
    #fig = plt.figure(0)
    #fig.canvas.set_window_title("Samplesize " , samplesize)
    #mng = plt.get_current_fig_manager()
    #mng.window.showMaximized()
    plt.show()

def mainfunction(datatrainx,datatrainy,datatestx, datatesty,dataones,datazeroes ,datatrainxactual, size):

     actualmse =[]
     for i in range(1,7):
         if ( i == 1 ):
             datatrainxnew = datazeroes
         elif (i== 2 ):
             datatrainxnew = dataones
         elif i == 3:
             datatrainxnew = datatrainxactual
         else:
             newvaluesarr =[]
             newvaluesarr = np.power(datatrainx,i-2)
             newarr = np.concatenate((np.array(datatrainlast),newvaluesarr),axis=2)
             datatrainxnew = newarr
         mse2 ,bias2,variance = calcbiasvariance(datatrainxnew, datatrainy,datatestx,datatesty , i, dataones, size)
         actualmse.append(mse2)
         print "    g",i,"(x)       ", "{:10.9f}".format(bias2) ,"      " ,"{:10.9f}".format(variance)
         datatrainlast = datatrainxnew
     plothist(actualmse,size)

def calcwl2 (pdtrainx,pdtrainy,eachlambda):
     aa=[]
     pdtrainxt = pdtrainx.T
     aa = pdtrainxt.dot(pdtrainx)
     Imatrix = np.identity(3)
     lambdamatrix = np.empty((3,3))
     lambdamatrix.fill(eachlambda)
     bb = np.multiply(lambdamatrix,Imatrix)
     sum1 = np.add(aa,bb)
     if sum1.size ==0:
             inverse = 0.0
     elif sum1.size == 1:
         inverse = np.reciprocal(sum1)
     else:
         inverse = np.linalg.pinv(sum1)
     a2 = np.dot(inverse, pdtrainxt)
     ans = np.dot(a2,pdtrainy)
     return ans

def  calcbiasvariancel2(datatrainx, datatrainy,datatestx,datatesty , index,eachlambda):
     #Linear regression without regularization
     mse = []
     hd =[]
     for eachrowx,eachrowy in zip(datatrainx,datatrainy):
        thishd =[]
        thishd =calcwl2(eachrowx, eachrowy,eachlambda)
        hd.append(thishd)

     #MSE on every dataset and draw
     mse1 = [mse[i].item() for i in range (0,len(mse))]
     bias2 =0.0
     variance = 0.0
     for eachrowx,eachrowy in zip(datatestx,datatesty):
         if ((index!=1) and (index!=2)):
             eachrowy = eachrowx
         aa = float(calceedhd(hd,index,eachrowx)) - float(eachrowy)
         bias2 = float(bias2) + float(aa*aa)
     for eachlinehd in hd:
        for eachrowx,eachrowy in zip(datatestx,datatesty):
             bb =  calchd(eachrowx,eachlinehd,index) - calceedhd(hd,index,eachrowx)
             variance = variance + (bb*bb)
     actualbias = float(bias2)/float(len(datatestx))
     actualvariance = float(variance)/float((len(datatestx)*len(hd)))
     return actualbias,actualvariance

def mainl2(datatrainx,datatrainy,datatestx, datatesty,dataones ,datatrainxactual, size,eachlambda):
     actualmse =[]
     for i in range(2,4):
         if i == 2:
             datatrainxnew = datatrainxactual
         else:
             newvaluesarr =[]
             newvaluesarr = np.power(datatrainx,i-2)
             newarr = np.concatenate((np.array(datatrainxnew),newvaluesarr),axis=2)
             datatrainxnew = newarr
     bias2,variance = calcbiasvariancel2(datatrainxnew, datatrainy,datatestx,datatesty , 4,eachlambda)
     print "  ","{:5.3f}".format(eachlambda),"        " , "{:10.9f}".format(bias2) ,"      " ,"{:10.9f}".format(variance)


def datapreprocess (datatrain):
     dftrainx = pd.DataFrame(datatrain['features'][:,18],  columns=["f19"])
     dftrainy = pd.DataFrame({'Y': value for value in datatrain['label']})
     dftrainy= dftrainy.replace(-1,0.0)

     for i in range ( 0 , 30):
         if i+1 == 2 or i+1 == 7 or i+1 == 8 or i+1 == 14 or i+1 == 15 or i+1 == 26 or i+1 == 29: #convert to 3 features
             data = datatrain['features'][:,i]
             newfeaturename__1 = "f"+str(i+1)+"a"
             newfeaturename_0 = "f"+str(i+1)+"b"
             newfeaturename_1 = "f"+str(i+1)+"c"
             data__1 =[]
             data_0 =[]
             data_1 = []
             for eachvalue in data:
                 if eachvalue == -1:
                     data__1.append(1.0)
                     data_0.append(0.0)
                     data_1.append(0.0)
                 elif eachvalue == 0:
                     data__1.append(0.0)
                     data_0.append(1.0)
                     data_1.append(0.0)
                 else:
                     data__1.append(0.0)
                     data_0.append(0.0)
                     data_1.append(1.0)
             dftrainx[newfeaturename__1] = data__1
             dftrainx[newfeaturename_0] = data_0
             dftrainx[newfeaturename_1] = data_1
         elif i+1 == 19: #Do nothing
             pass
         else:  #convert to 2 features
             data = datatrain['features'][:,i]
             newfeaturename = "f"+str(i+1)
             data_1 =[]
             for eachvalue in data:
                 if eachvalue == -1:
                     data_1.append(0.0)
                 else:
                     data_1.append(1.0)
             dftrainx[newfeaturename] = data_1
     return dftrainx, dftrainy

if __name__ == "__main__":
     indices = ["0","1","2","3","4","5","6","7","8","9"]
     # BIAS VARIANCE TRADE-OFF
     # Generate 100 datasets each set with 10 data points
     print " "
     print "1.(a)"
     print "    FUNCTION        BIAS2                VARIANCE"
     #GET DATA
     datatrainx=[]
     datatrainy=[]
     for i in range (0, 100):
         arrx=[]
         for j in range (0,10):
             arrx.append([random.uniform(-1, 1) ])
         datatrainx.append(np.asarray (arrx,order=1))
         arry=[]
         for eachxarr in datatrainx[i]:
              arry.append([getyforx(eachxarr[0])])
         datatrainy.append(np.asarray(arry, dtype='float64'))

     datatestx=[]
     datatesty=[]
     for i in range (0, 500):
         arrx=[]
         arrx = np.random.uniform(low=-1, high=1, size=1)
         datatestx.append(np.asarray(arrx, dtype='float64'))
         arry=[]
         for eachx in datatestx[i]:
              arry.append(getyforxwithoutnoise(eachx))
         datatesty.append(np.asarray(arry, dtype='float64'))
     #add one to the arr
     datatrainxactual =[]
     dataones =[]
     datazeroes = []
     for eachline in datatrainx:  #add features as neccesary
        newarr = np.insert(eachline,1,1.0,axis=1)
        dataones.append(np.ones((10,1)))
        datazeroes.append(np.zeros((10,1)))
        datatrainxactual.append(newarr)

     mainfunction(datatrainx,datatrainy,datatestx, datatesty  , dataones,datazeroes ,datatrainxactual,10)

     print " "
     print "1.(b)"
     print "    FUNCTION        BIAS2                VARIANCE"
     datatrainx=[]
     datatrainy=[]
     for i in range (0, 100):
         arrx=[]
         for j in range (0,100):
             #arrx.append([random.uniform(-1, 1) , 1.0])
             arrx.append([random.uniform(-1, 1) ])
         datatrainx.append(np.asarray (arrx,order=1))
         arry=[]
         for eachxarr in datatrainx[i]:
              arry.append([getyforx(eachxarr[0])])
         datatrainy.append(np.asarray(arry, dtype='float64'))
          #add one to the arr
     datatrainxactual =[]
     dataones =[]
     datazeroes =[]
     for eachline in datatrainx:  #add features as neccesary
        newarr = np.insert(eachline,1,1.0,axis=1)
        dataones.append(np.ones((100,1)))
        datazeroes.append(np.zeros((100,1)))
        datatrainxactual.append(newarr)

     mainfunction(datatrainx,datatrainy,datatestx, datatesty, dataones,datazeroes ,datatrainxactual,100 )

     print " "
     print "1.(d) g4(x)=w0 + w1x + w2x2 , 100 samples"
     print "    Lambda        BIAS2                VARIANCE"
     lambdavalues = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0,10]
     for eachlambda in lambdavalues:
         mainl2(datatrainx,datatrainy,datatestx, datatesty, dataones ,datatrainxactual,100,eachlambda)


     #LINEAR AND KERNEL SVM
     print " "
     print "2.(a) LINEAR SVM IN LIBSVM"
     print "       Cvalue      Crossvalidation accuracy       Average Training time"
     datatrain = io.loadmat('phishing-train.mat')
     datatest = io.loadmat('phishing-test.mat')
     #Data preprocessing
     dftrainx,dftrainy = datapreprocess (datatrain)
     dftestx,dftesty = datapreprocess (datatest)

     #SVM train
     cvalues = [-6,-5,-4,-3,-2,-1,0,1,2]
     for eachc in cvalues:
         cvaccuracy =0.0
         avgtrainingtime = 0.0
         eachc = math.pow(4,eachc)
         param = svm_parameter('-q -t 0 -v 3 -c ' + str(eachc))
         classes = dftrainy["Y"].values
         data =dftrainx.values.tolist()
         #classes = classes.tolist()
         #data = data.tolist()
         # formulate as libsvm problem
         problem = svm_problem(classes, data)
         current_time1 = datetime.now().time()
         model = svm_train(problem, param)
         current_time2 = datetime.now().time()
         cvaccuracy = model

         avgtrainingtime = (datetime.combine(date.today(), current_time2) - datetime.combine(date.today(), current_time1))/3
         print "      " , "{:10.6f}".format(eachc),"     ",format(cvaccuracy,'.2f'),"                     ",avgtrainingtime

     print " "
     print "2.(b) KERNEL SVM IN LIBSVM"
     print "      Polynomial Kernel"
     print "       Cvalue       Degree  Crossvalidation accuracy       Average Training time"
     cvalues = [-3,-2,-1,0,1,2,3,4,5,6,7]
     degree  = [1,2,3]
     for eachc in cvalues:
         cvaccuracy =0.0
         avgtrainingtime = 0.0
         eachc = math.pow(4,eachc)
         for eachdegree in degree:
             param = svm_parameter('-q -t 1 -v 3 -c ' + str(eachc)+" -d "+ str(eachdegree))
             current_time1 = datetime.now().time()
             cvaccuracy = svm_train(problem, param)
             current_time2 = datetime.now().time()
             avgtrainingtime = (datetime.combine(date.today(), current_time2) - datetime.combine(date.today(), current_time1))/3
             print "      " ,  "{:10.4f}".format(eachc),"     ",eachdegree, "      ",format(cvaccuracy,'.2f'),"                     ",avgtrainingtime

     print " "
     print "      RBF Kernel"
     print "          Cvalue       Gamma        Crossvalidation accuracy       Average Training time"
     cvalues = [-3,-2,-1,0,1,2,3,4,5,6,7]
     gamma  = [-7,-6,-5,-4,-3,-2,-1]
     for eachc in cvalues:
         cvaccuracy =0.0
         avgtrainingtime = 0.0
         eachc = math.pow(4,eachc)
         for eachgamma in gamma:
             eachgamma = math.pow(4,eachgamma)
             param = svm_parameter('-q -t 2 -v 3 -c ' + str(eachc)+" -g "+ str(eachgamma))
             current_time1 = datetime.now().time()
             cvaccuracy = svm_train(problem, param)
             current_time2 = datetime.now().time()
             avgtrainingtime = (datetime.combine(date.today(), current_time2) - datetime.combine(date.today(), current_time1))/3
             print "      " ,  "{:10.4f}".format(eachc),"     ","{:8.7f}".format(eachgamma), "      ",format(cvaccuracy,'.2f'),"                     ",avgtrainingtime

