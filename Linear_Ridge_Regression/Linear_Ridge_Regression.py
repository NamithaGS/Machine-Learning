from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import math

dataaxes ={
'CRIM': ['per capita crime rate by town ', 'Median value of owner-occupied homes in $1000s'],
'ZN': ['proportion of residential land zoned for lots over 25,000 sq.ft','land'],
'INDUS':['proportion of non-retail business acres per town ','Number of towns'],
'CHAS':['Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) ' ,''],
'NOX': ['nitric oxides concentration (parts per 10 million)' ,''],
'RM': ['average number of rooms per dwelling ',''],
'AGE': ['proportion of owner-occupied units built prior to 1940 ',''],
'DIS': ['weighted distances to five Boston employment centres' ,''],
'RAD': ['index of accessibility to radial highways ',''],
'TAX': ['full-value property-tax rate per $10,000 ',''],
'PTRATIO': ['pupil-teacher ratio by town ',''],
'B': ['1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town ',''],
'LSTAT': ['% lower status of the population ',''],
'MEDV': ['Median value of owner-occupied homes in $1000s','']
}

def dataset_to_dataframe(traindatax,traindatay ,  featurenames):
    dfx = pd.DataFrame(traindatax,  columns=featurenames)
    dfx ["MEDV"] = traindatay
    return dfx

def calculate_pearson_and_draw(dftrain,boston):
    pcor = dftrain.corr(method='pearson')
    print "Pearsons Coefficient : \n" , pcor["MEDV"].drop("additional")
    i=0

    for eachattribute in boston.feature_names:
        attr = dftrain[eachattribute]
        plt.subplot(2,3,i+1)
        plt.title(eachattribute)
        plt.ylabel("Frequency")
        plt.xlabel(dataaxes[eachattribute][0])
        plt.hist(attr,bins=10)

        if (i==5 ):
            mng = plt.get_current_fig_manager()

            mng.window.showMaximized()
            plt.show()
            i=0
        else:
            i=i+1

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    return pcor

def add1(x_train_std , y_train_std):
     ax = np.insert(np.array(x_train_std), 0,1.0,axis=1)
     return ax,y_train_std

def standardize_add1(x,y,x_train , y_train):
    z_scores_np_x_ = (pd.DataFrame( x) - pd.DataFrame(x_train).mean()) / pd.DataFrame(x_train).std()
    z_scores_np_x = np.asarray(z_scores_np_x_)
    ax = np.insert(np.array(z_scores_np_x), 0,1.0,axis=1)
    return ax, y

def standardize_add1pd(  pdtrainxfe , pdtestxfe):
    z_scores_np_x_train = (pd.DataFrame( pdtrainxfe) - pd.DataFrame(pdtrainxfe).mean()) / pd.DataFrame(pdtrainxfe).std()
    z_scores_np_x_test = (pd.DataFrame( pdtestxfe) - pd.DataFrame(pdtrainxfe).mean()) / pd.DataFrame(pdtrainxfe).std()
    z_scores_np_x_train = np.insert(np.array(z_scores_np_x_train), 0,1.0,axis=1)
    z_scores_np_x_test = np.insert(np.array(z_scores_np_x_test), 0,1.0,axis=1)
    #z_scores_np_x_train["additional"] = np.array(1)
    #z_scores_np_x_test["additional"] = np.array(1)
    return z_scores_np_x_train, z_scores_np_x_test

def  msecalc(ytrue,ypred):
    diff =0.0
    N = len(ytrue)
    for i in range(0,N):
        a = ytrue[i]
        b = ypred[i]
        diff1 = a - b
        diff = diff +  pow(diff1,2)
    return diff/float(N)

def residuecalc(ytrue,ypred):
    diff = 0.0
    N = len(ytrue)
    for i in range(0,N):
        a = ytrue[i]
        b = ypred[0][i]
        diff = diff +( a- b)
    return diff

def calcb (pdtrainx,pdtrainy):
     pdtrainxt = pdtrainx.T
     a1 = pdtrainxt.dot(pdtrainx)
     inverse = np.linalg.pinv(a1)
     a2 = np.dot(inverse, pdtrainxt)
     ans = np.dot(a2,pdtrainy)
     return ans

def linearregression(pdtrainx, pdtrainy,pdtestx, pdtesty, y_train_std, y_test_std ):
     b = calcb (pdtrainx,pdtrainy )
     pdtestxt = pdtestx.T
     pdtrainxt = pdtrainx.T
     ypredtrain= np.dot(b.T,pdtrainxt)
     ypredtest= np.dot(b.T,pdtestxt)
     return ypredtrain[0], ypredtest[0]

def calcbforridge (pdtrainx,pdtrainy,eachlambda):
     pdtrainxt = pdtrainx.T
     aa = pdtrainxt.dot(pdtrainx)
     Imatrix = np.identity(14)
     lambdamatrix = np.empty((14,14))
     lambdamatrix.fill(eachlambda)
     bb = np.multiply(lambdamatrix,Imatrix)
     #bb = np.fill_diagonal(Imatrix.values,eachlambda)
     inverse = np.linalg.pinv(np.add(aa,bb))
     a2 = np.dot(inverse, pdtrainxt)
     ans = np.dot(a2,pdtrainy)
     return ans

def ridgeregression(pdtrainx, pdtrainy,pdtestx, pdtesty, y_train_std, y_test_std,lambdaa):
     for eachlambda in lambdaa:
         b = calcbforridge (pdtrainx,pdtrainy,eachlambda )
         pdtestxt = pdtestx.T
         pdtrainxt = pdtrainx.T
         ypredtrain= np.dot(b.T,pdtrainxt)
         ypredtest= np.dot(b.T,pdtestxt)
         aa = np.square(b)
         w22 = np.sum(aa)
         #mse = ((msecalc(y_train_std,ypredtrain[0]) * len(y_train)) + (eachlambda  *math.sqrt( w22)))/len(y_train)
         mse = msecalc(y_train_std,ypredtrain[0])
         print "MSE for lambda training set: " , eachlambda , " = " , mse
         aa = np.square(b)
         w22 = np.sum(aa)
         #mse = ((msecalc(y_test_std,ypredtest[0]) * len(y_train) )+ (eachlambda  * math.sqrt(w22)))/len(y_train)
         mse = msecalc(y_test_std,ypredtest[0])
         print "MSE for lambda test set    : " , eachlambda , " = " , mse

def ridgeregressioncv(foldnumber,pdtrain11,pdtestx,y_test_std, boston ,lambdaa):
    mseperfold =[]
    for eachlambda in lambdaa:
         pdtrainperfold =pd.DataFrame.sample( pdtrain11, frac=0.9,replace=False,random_state=2)
         rows = pdtrainperfold.index.values
         pdtestperfold = pdtrain11.drop(rows)
         pdtrainyperfold = pdtrainperfold["MEDV"]
         pdtrainxperfold = pdtrainperfold.drop("MEDV",axis=1)
         pdtestyperfold =  pdtestperfold["MEDV"]
         pdtestxperfold =  pdtestperfold.drop("MEDV",axis=1)

        #get random set of 9/10th data of trainx
         b = calcbforridge (pdtrainxperfold,pdtrainyperfold ,eachlambda)
         pdtestxt = pdtestxperfold.T
         pdtrainxt = pdtrainxperfold.T
         ypredtrain= np.dot(b.T,pdtrainxt)
         ypredtest= np.dot(b.T,pdtestxt)
         aa = np.square(b)
         w22 = np.sum(aa)
         mse = msecalc(pdtestyperfold.values,ypredtest)
         #mse = msecalc(pdtestyperfold.values,ypredtest) + (eachlambda  * math.sqrt(w22))
         #mse = (( msecalc(pdtestyperfold.values,ypredtest) * len(pdtestyperfold.values) ) + (eachlambda  * math.sqrt(w22)))/len(pdtestyperfold.values)
         #print "MSE for lambda test set    : " , eachlambda , " = " , mse
         mseperfold.append(mse)
    return mseperfold

def selecttop4features(pcor):
    aa = pcor["MEDV"]
    bb = aa.drop(["MEDV","additional"])
    cc =  pd.Series.abs(bb)
    dd = pd.Series.sort_values(cc)
    #print dd
    max = dd[-4:]
    maxdp = pd.DataFrame(max)
    return maxdp

def listofallcombioffeaturenamescalc(boston):
    list1 = []
    everything = itertools.combinations(boston.feature_names, 4)
    for eachthing in everything:
        list1.append(list(eachthing))
    return list1


if __name__ == "__main__":
     boston = load_boston()
     #The test set consists of all (7i)-th data points (where i = 0; 1; 2; : : : ; 72),
     # This will create a test set of size 73 and a training set of size 433.
     alldatax, alldatay = boston.data , boston.target
     x_test =[]
     y_test=[]
     x_train=[]
     y_train =[]
     #Splitting

     for i in range (0,len(alldatax)):
          if(( i %7)==0):
               x_test.append(alldatax[i])
               y_test.append(alldatay[i])
          else:
               x_train.append(alldatax[i])
               y_train.append(alldatay[i])

     #turn to pandas dataframe
     newfeaturenames =[]
     newfeaturenames.append("additional")
     for eachname in boston.feature_names:
         newfeaturenames.append(eachname)
     dftrainx = pd.DataFrame(x_train, columns=boston.feature_names)
     dftrainy =  pd.DataFrame(y_train, columns=["MEDV"])
     dftrain = dataset_to_dataframe(x_train, y_train , boston.feature_names)
     dftestx = pd.DataFrame(x_test, columns=boston.feature_names)
     dftesty =  pd.DataFrame(y_test, columns=["MEDV"])
     dftest = dataset_to_dataframe(x_test, y_test , boston.feature_names)

     #pcor = calculate_pearson_and_draw(dftrain,boston)

     x_train_std,y_train_std = standardize_add1(x_train , y_train,x_train , y_train)
     x_test_std,y_test_std = standardize_add1(x_test , y_test,x_train , y_train)

     pdtrainx = pd.DataFrame(x_train_std,columns=newfeaturenames)
     pdtrainy = pd.DataFrame(y_train_std,columns=["MEDV"])
     pdtestx = pd.DataFrame(x_test_std,columns=newfeaturenames)
     pdtesty = pd.DataFrame(y_test_std,columns=["MEDV"])
     pdtrain = pd.concat([pdtrainx,pdtrainy],axis=1)

     pcor = calculate_pearson_and_draw(pdtrain,boston)

     # LINEAR REGRESSION
     print ("\nLINEAR REGRESSION ")
     ypredtrain,ypredtest =  linearregression(pdtrainx, pdtrainy,pdtestx, pdtesty, y_train_std, y_test_std )
     print "MSE of training data set : "  , msecalc(y_train_std,ypredtrain)
     print "MSE of testing data set : "  , msecalc(y_test_std,ypredtest)

     #RIDGE REGRESSION
     print ("\nRIDGE REGRESSION ")
     lambdaa =[0.01,0.1,1.0]
     ridgeregression(pdtrainx, pdtrainy,pdtestx, pdtesty, y_train_std, y_test_std,lambdaa)

     #RIDGE REGRESSION cross validation
     print ("\nRIDGE REGRESSION with CROSS VALIDATION")
     step = 0.5
     i = 0.0001
     lambdaa=[]
     while i<=10:
         lambdaa.append(i)
         i = i+step
     #lambdaa =[0.0001,0.001,0.01,0.1,1,10]
     mse={}
     for i in range (0,10):
         #print "CV FOLD " , i
         mse[i] = ridgeregressioncv(i,pdtrain,pdtestx,y_test_std, boston ,lambdaa)
     #best lambda?
     #print mse
     lambdamse=[0.0] * len(lambdaa)
     for eachi,eachline in mse.items():
         for i in range (0 , len(lambdaa)):
            lambdamse[i] = lambdamse[i] + eachline[i]
     lambdamse1 =[]
     for eachvalue in lambdamse:
         lambdamse1.append (eachvalue/float(len(mse)))
     i = 0
     for eachvalue in lambdamse1:
         print "Lambda : " , lambdaa[i],". MSE : " ,eachvalue
         i = i +1
     minvalue = min(lambdamse)
     bestlambda = lambdamse.index(minvalue)

     print "Best Lamba value :  " , lambdaa[bestlambda]
     ridgeregression (pdtrainx, pdtrainy,pdtestx, pdtesty, y_train_std, y_test_std,[lambdaa[bestlambda]])

     #FEATURE SELECTION
     print "\nFEATURE SELECTION "
     print "1.SELECTION  with CORRELATION"
     top4featurespseries = selecttop4features(pcor)

     #print top4featurespseries
     top4featurenames = top4featurespseries.index.values
     top4featurenames=np.append(top4featurenames,"additional")
     pdtrainxcv = pd.DataFrame(pdtrainx[top4featurenames], columns=[top4featurenames])
     pdtestxcv = pd.DataFrame(pdtestx[top4featurenames], columns=[top4featurenames])
     ypredtraincv,ypredtestcv =  linearregression(pdtrainxcv, pdtrainy,pdtestxcv, pdtesty, y_train_std, y_test_std )
     top4featurenames = np.delete(top4featurenames,np.argwhere(top4featurenames=="additional"))
     print "    a ) Best features : ", top4featurenames
     print "   " , top4featurespseries
     print "        MSE on training data with the best features :", msecalc(y_train_std,ypredtraincv)
     print "        MSE on testing data with the best features :", msecalc(y_test_std,ypredtestcv)

     #print top4featurespseries
     topfeaturename = top4featurespseries.index[-1:].values
     #print "top feature : ", topfeaturename
     topfeaturesdf = pdtrainx.drop(topfeaturename,axis=1)
     top4featurenamesend = []
     top4featurenamesend.append(topfeaturename.item())
     lala = []
     print "    b )"
     print "        Selected feature :   ",topfeaturename, ". Correlation :", top4featurespseries[-1:].values
     while len(top4featurenamesend) <=3 :
         lala.append(topfeaturename.item())
         lala.append("additional")
         #pdtrainxcv = pd.DataFrame(topfeature.values, columns=[topfeature.idxmax()])
         pdtrainxcv = pd.DataFrame(pdtrainx[lala], columns=[lala])
         pdtestxcv = pd.DataFrame(pdtestx[lala], columns=[lala])
         ypredtraincv,ypredtestcv =  linearregression(pdtrainxcv, pdtrainy,pdtestxcv, pdtesty, y_train_std, y_test_std )
         yresd = y_train_std - ypredtraincv[0]
         yresddf = pd.DataFrame(yresd,columns=["MEDV"])
         newfeaturelisttouse = pdtrainx[topfeaturesdf.columns.values]
         newfeaturelisttouse.is_copy = False  #remove spurious warning
         newfeaturelisttouse["MEDV"] = yresd
         pcor = newfeaturelisttouse.corr(method='pearson')

         top4featurespseries = selecttop4features(pcor)
         #print top4featurespseries
         topfeaturename = top4featurespseries.index[-1:].values
         print "        Selected feature :   ",topfeaturename, ".   Correlation :", top4featurespseries[-1:].values
         top4featurenamesend.append(topfeaturename.item())
         #print "top feature : ", topfeaturename
         topfeaturesdf = topfeaturesdf.drop(topfeaturename,axis=1)

     print "        Best features : ", top4featurenamesend
     ypredtraincv,ypredtestcv =  linearregression(pdtrainxcv, pdtrainy,pdtestxcv, pdtesty, y_train_std, y_test_std )
     print "        MSE on training test with the best features :", msecalc(y_train_std,ypredtraincv)
     print "        MSE on testing test with the best features :", msecalc(y_test_std,ypredtestcv)

     print "\n2.SELECTION  with BRUTE FORCE"
     listofallcombioffeaturenames = listofallcombioffeaturenamescalc(boston)
     msetrainmin = sys.maxint
     msetestmin = sys.maxint
     for each4featurecombi in listofallcombioffeaturenames:
         each4featurecombi = np.append(each4featurecombi,"additional")
         pdtrainxbrute = pd.DataFrame(pdtrainx[each4featurecombi], columns=[each4featurecombi])
         pdtestxbrute = pd.DataFrame(pdtestx[each4featurecombi], columns=[each4featurecombi])
         ypredtrainbrute,ypredtestbrute = linearregression(pdtrainxbrute, pdtrainy,pdtestxbrute, pdtesty, y_train_std, y_test_std )
         msetrain = msecalc(y_train_std,ypredtrainbrute)
         msetest = msecalc(y_test_std,ypredtestbrute)
         if msetrain <msetrainmin:
             msetrainmin = msetrain
             bestfeaturetrain = each4featurecombi
             msetestmin = msetest
             #bestfeaturetest = each4featurecombi
         # if msetest <msetestmin:
         #     msetestmin = msetest
         #     bestfeaturetest = each4featurecombi
     x = np.array(bestfeaturetrain)
     index = np.argwhere(x=="additional")
     bestfeaturetrain=  np.delete( bestfeaturetrain,index )
     x = np.array(bestfeaturetrain)
     index = np.argwhere(x=="additional")
     bestfeaturetest= np.delete( bestfeaturetrain, index)


     print "    Best features on Training data : ", bestfeaturetrain , ". MSE : " , msetrainmin
     print "    Best features on Testing data  : ", bestfeaturetest , ". MSE : " , msetestmin

     #POLYNOMIAL FEATURE EXPANSION
     #get all combi of the feature names
     allfeaturenames = boston.feature_names
     allcombifeaturenames = []
     for subset in itertools.combinations(allfeaturenames, 2):
        allcombifeaturenames.append(subset)
     for eachname in boston.feature_names:
         allcombifeaturenames.append((eachname,eachname))
     pdtrainxfe = pdtrainx.drop("additional",axis=1)
     pdtestxfe = pdtestx.drop("additional",axis=1)
     for eachtuple in allcombifeaturenames:
         newdata = pdtrainx[eachtuple[0]].values * pdtrainx[eachtuple[1]].values
         newcolname = eachtuple[0]+"_"+eachtuple[1]
         pdtrainxfe[newcolname] = newdata

         newtestdata = pdtestx[eachtuple[0]].values * pdtestx[eachtuple[1]].values
         newtestcolname = eachtuple[0]+"_"+eachtuple[1]
         pdtestxfe[newtestcolname] = newtestdata

     pdtrainxfen,pdtestxfen = standardize_add1pd( pdtrainxfe , pdtestxfe)

     ypredtrain,ypredtest =  linearregression(pdtrainxfen, pdtrainy,pdtestxfen, pdtesty, y_train_std, y_test_std )
     print "\n3.POLYNOMIAL FEATURE EXPANSION"
     print "    MSE of training data set : "  , msecalc(y_train_std,ypredtrain)
     print "    MSE of testing data set : "  , msecalc(y_test_std,ypredtest)








