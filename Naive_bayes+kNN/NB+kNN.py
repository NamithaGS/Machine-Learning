import json
import math
import collections
import sys
def parsefile(text):
    datadict1 = []
    for line in text:
        datadict = {}
        data=line.split(',')
        datadict["id"] = data[0]
        datadict["ri"] = data[1]
        datadict["na"] = data[2]
        datadict["mg"] = data[3]
        datadict["al"] = data[4]
        datadict["si"] = data[5]
        datadict["k"] = data[6]
        datadict["ca"] = data[7]
        datadict["ba"] = data[8]
        datadict["fe"] = data[9]
        datadict["class"] = data[10]
        datadict1.append(datadict)
    return datadict1

####### NAIVE BAYES##################
def priorprobfunc(text):
    prorprob = {}
    for eachline in text:
        classi = eachline["class"]
        if prorprob.has_key(classi):
            prorprob[classi] = prorprob[classi]+ 1
        else:
            prorprob[classi] = 1
    count = 0
    for eachcount in prorprob.values():
        count = count + eachcount
    prorprob1 ={}
    for key,value in prorprob.items():
        prorprob1[key]= float (value/ float(count))
    return prorprob1

def gaussiancalc (x,var3,mu3):
    xu = math.pow(x-mu3 ,2)
    if var3 == 0.0:
        return 0
    else:
        aa1 =  math.exp((( 0.0 - float( xu))/ (2.0 * var3 )) )
        aa2 =  1.0 / float( math.sqrt(2.0*3.14159* var3) )
        return  aa2 *  aa1

def probclassgivenfeature(datadict):
    #get sum of all data in a class into P
    P= {}
    for eachline in datadict:
        classi = eachline["class"]
        if P.has_key(classi):
            classidict = P.get(classi)
            thisclassperfeaturecounter={}
            for key,value in classidict.items():
                thisclassperfeaturecounter[key] = float(value) + float(eachline[key])
            P[classi]= thisclassperfeaturecounter
        else:
            P[classi]= eachline

    #get n of each class each feature
    n={}
    for eachclass,dictcount in P.items():
        for key, value in dictcount.items():
            if key=="class":
                newclass = float(value)/float(eachclass)
                P[eachclass]["class"] = newclass  #contains count of each class, needed for average
                n[eachclass] = newclass


    ##calculate mu
    mu = {}
    for eachclass,dictcount in P.items():
        newdictcountavg = {}
        for key, value in dictcount.items():
            newdictcountavg[key] =  float(value) / float(n[eachclass])
        mu[eachclass] =  newdictcountavg

    ##Calucate variance of each class each feature
    newdictcountsumxu2 = {}
    for eachline in datadict:
        classi = eachline["class"]
        newdictcountsumxu2local = {}
        for eachfeature,value in eachline.items():
            if(newdictcountsumxu2.has_key(classi)):
                newdictcountsumxu2local[eachfeature] = newdictcountsumxu2[classi][eachfeature] + math.pow(float(value) - float(mu[classi][eachfeature]) , 2)
            else:
                newdictcountsumxu2local[eachfeature] = math.pow(float(value) - float(mu[classi][eachfeature]) , 2)
        newdictcountsumxu2[classi] =  newdictcountsumxu2local
    var ={}
    for eachclass,dictcount in newdictcountsumxu2.items():
        temp = {}
        for key,value in dictcount.items():
            temp1 = float(value)/(float(n[eachclass]) -1)
            #if(temp1==0.0): temp1 = 1.0
            temp[key]= temp1

        var[eachclass] =  temp

    ##Calculate PDF function for each class each feature
    pdf ={}
    for eachline in datadict:
        temppdf = {}
        classi = eachline["class"]
        for feature,value in eachline.items():
            vartemp = float ( var[classi][feature] )
            mutemp =  (mu[classi][feature])
            x = float (value)
            if pdf.has_key(classi):
                temppdf[feature] = pdf[classi][feature] * gaussiancalc(x,vartemp,mutemp)
            else:
                temppdf[feature] = gaussiancalc(x,vartemp,mutemp)
        pdf[classi] = temppdf
    return pdf,var,mu

def classifyNB(datadict, priorprob , P, allclasses,var1,mu1):
     classes = []
     #Calculate product of prob of class given features and prior prob

     for eachline in datadict:
         PostP={}#for every test data
         for eachclass in allclasses:  #calculate for all classes:
             totalpclassgivenfeature = 1.0
             for key, value in eachline.items():
                 if(key!="class") and key!="id":
                     aa = gaussiancalc (float(value),var1[eachclass][key],mu1[eachclass][key])
                     totalpclassgivenfeature = totalpclassgivenfeature * aa

             PostP[eachclass] = totalpclassgivenfeature  * float(priorprob[eachclass])

         #Arg max of all classes
         max = -sys.maxint
         argmaxclass =1
         for key,value in PostP.items():
             if value  > float(max):
                argmaxclass = key
                max = value
         classes.append(argmaxclass)
     return classes

def checkaccuracy( correctclassification, ourclassification):
    correct = 0
    total =0
    for i in range ( 0, len(correctclassification)):
        if( correctclassification[i] == ourclassification[i]):
            correct+=1
        total+=1
    #print( "correct : " , correct)
    #print("total : " ,total)
    return float(correct) /float(total)

def muperfeaturecalc(datadictraining,eachfeature):
    muperfeature={}
    for eachline in datadictraining:
        for eachfeature, value in eachline.items():
            if muperfeature.has_key(eachfeature):
                temp = muperfeature[eachfeature]
                muperfeature[eachfeature] = float(temp) + float(value)
            else:
                muperfeature[eachfeature] = float(value)
    for eachfeature,eachvaluesum in muperfeature.items():
        muperfeature[eachfeature] = eachvaluesum/len(datadictraining)
    return muperfeature

def varsdperfeaturecalc(datadictraining,muperfeature,eachfeature):
    sigmaperfeature ={}
    sigmaperfeaturenum={}
    for eachline in datadictraining:
        for eachfeature,value in eachline.items():
            if sigmaperfeaturenum.has_key(eachfeature):
                sigmaperfeaturenum[eachfeature] = float(sigmaperfeaturenum[eachfeature]) +math.pow((float(value)- float(muperfeature[eachfeature])),2)
            else:
                sigmaperfeaturenum[eachfeature] = math.pow(float(value)- float(muperfeature[eachfeature]),2)

    for eachfeature,eachvaluesum in sigmaperfeaturenum.items():
        sigmaperfeature[eachfeature] = math.sqrt(float(eachvaluesum)/ (len(datadictraining)-1))
    return sigmaperfeature

def getnormalizedvalues(data,mu,sigma):
    normalizedvalues = {}
    i = 0
    for eachline in data:
        templist ={}
        for eachfeature,value in eachline.items():
            x = float(value)
            templist[eachfeature] = ((x - mu[eachfeature])/sigma[eachfeature])
        normalizedvalues[i] = templist
        i=i+1
    return normalizedvalues

def calculateL1(normalizedtestingperfeatureperline, normalizedtrainingperfeature,train):
    L1count ={}
    for eachid,eachline in normalizedtrainingperfeature.items():
        L1temp = 0.0
        for eachfeature, valuexntrain in eachline.items():
            if( eachfeature!="class") and (eachfeature!="id"):
                valuexntest = normalizedtestingperfeatureperline[eachfeature]
                L1temp  = L1temp+ abs(  float(valuexntest) - float(valuexntrain))
        L1count[eachid] = L1temp
    return L1count


def calculateL2(normalizedtestingperfeatureperline, normalizedtrainingperfeature,train):
    L2count ={}
    for eachid,eachline in normalizedtrainingperfeature.items():
        L1temp = 0.0
        for eachfeature, valuexntrain in eachline.items():
            if( eachfeature!="class") and (eachfeature!="id"):
                valuexntest = normalizedtestingperfeatureperline[eachfeature]
                L1temp  = L1temp+ pow(abs( float(valuexntest) - float(valuexntrain)),2)
        L2count[eachid] = L1temp

    #take sqrt of all values
    L2count1={}
    for eachlineid, eachcount in L2count.items():
        L2count1[eachlineid] = math.sqrt(float(eachcount))
    return L2count


def gettopvalues(datadictraining,Lperfeature,eachk,train):
    #for each line in L1perfeature get min k values
    listoftopclasses={}
    for eachid1 ,eachline in Lperfeature.items():

        s = sorted(eachline.iteritems(), key=lambda x:x[1])[:eachk]
        if s[0][1]== 0.0 and train ==1 :
            t = sorted(eachline.iteritems(), key=lambda x:x[1])[1:eachk+1]
        else: t = s
        listoftopidvalue ={}
        for items in t:
            listoftopidvalue[items[0]] =items[1]
        #get what the id is pointing to
        listoftopclassesperline=[]
        for eachid in listoftopidvalue.items():
            for eachfeature,value in datadictraining[eachid[0]].items():
                if eachfeature=="class":
                    listoftopclassesperline.append((value,eachid[1]))
        listoftopclasses[eachid1] = listoftopclassesperline
    return listoftopclasses

def most_common(lst):
    return max((lst), key=lst.count)

def resolvetieandgetmaxclass (topkvaluesL1, eachk):
    topkvaluesL1new={}
    if eachk==1 :
        for eachid, eachline in topkvaluesL1.items():
            topkvaluesL1new[eachid] = eachline[0]
        # same distance?
        return topkvaluesL1new
    else:

        #find the frequency of each class and return the max frequency ones
        for eachid, eachline in topkvaluesL1.items():
            #for this id get the tuple in a list
            topclassdict = []
            thisedtuplelist = []
            for eachtuple in eachline:
                thisedtuplelist.append(eachtuple[0])



            counter=collections.Counter(thisedtuplelist)
            aa = most_common(thisedtuplelist)
            topclasstuplecount = counter.most_common(1)

            m = max(v for _, v in counter.iteritems())         # get max frq
            r = [eachk for eachk, v in counter.iteritems() if v == m]   # r contains all classes with the highest freq

            thetopclass = topclasstuplecount[0][0]
            mymaxclass = 0
            mymaxdist =0
            if len(r)==1:
                #there is no contention in freq
                mymaxclass = r[0]
            else:
                #there is contention in freq, calc dist

                count = {}
                for eachclass in r:
                    counttemp = 0.0
                    for eachidtuple in eachline:
                        if eachidtuple[0] == eachclass:
                            counttemp=  counttemp + eachidtuple[1]
                    count[eachclass] = counttemp

                mymaxclass = min(count, key=count.get)

            for eachtuple in eachline:
                if eachtuple[0] == mymaxclass:
                    mymaxdist = mymaxdist + eachtuple[1]
            topclassdict.append((mymaxclass, mymaxdist))
            topkvaluesL1new[eachid] =mymaxclass
        return topkvaluesL1new

def classifyKNN(datadictraining, datadicttesting, muperfeature , sigmaperfeature,eachk, train):
    #Normalize all the training values per feature
    normalizedtrainingperfeature = getnormalizedvalues(datadictraining,muperfeature,sigmaperfeature)
    normalizedtestingperfeature = getnormalizedvalues(datadicttesting,muperfeature,sigmaperfeature)
    L1pertestingline={}
    L2pertestingline={}
    for eachid,eachline in normalizedtestingperfeature.items():
        L1pertestingline[eachid] = calculateL1(eachline, normalizedtrainingperfeature, train)
        L2pertestingline[eachid] = calculateL2(eachline, normalizedtrainingperfeature, train)

    #Take top k classes from L1 and L2
    topkvaluesL1 = gettopvalues(datadictraining,L1pertestingline,eachk,train)
    topkvaluesL2 = gettopvalues(datadictraining,L2pertestingline,eachk,train)

    # get the max item resolving tie s
    topvalueL1 = resolvetieandgetmaxclass (topkvaluesL1, eachk)
    topvalueL2 = resolvetieandgetmaxclass (topkvaluesL2, eachk)

    topclassL1 =[]
    topclassL2 =[]
    for classid,eachlne in topvalueL1.items():
        topclassL1.append(eachlne[0])
    for classid,eachlne in topvalueL2.items():
        topclassL2.append(eachlne[0])
    return topclassL1, topclassL2


if __name__ == "__main__":
     filenametrain = "train.txt"
     fp = open(filenametrain, "rb")
     texttrain = fp.read().split()
     datadictraining = parsefile(texttrain)
     filenametest = "test.txt"
     fp1 = open(filenametest, "rb")
     texttest = fp1.read().split()
     datadicttest = parsefile(texttest)

     ##### NAIVE BAYES #####
     ######## MODELLING #######
     #get prior prob of all class
     print "#### NAIVE BAYES Classification ####"
     priorprob = priorprobfunc(datadictraining)
     #Calculate Prob of all classes given features
     #  P ( Class | f1) , P (Class|f2)
     #gaussian distribution
     P,var2,mu2 = probclassgivenfeature(datadictraining)
     allclasses=[]
     for eachclass,value in P.items():
         allclasses.append(eachclass)
     ###### CLASSIFY #########
     #Accuracy on training data
     correctclassification= []
     for eachline in datadictraining:
        correctclassification.append(eachline["class"])
     ourclassification = classifyNB(datadictraining, priorprob , P, allclasses,var2,mu2)
     accuracy = checkaccuracy(correctclassification, ourclassification)
     print "TRAINING ACCURACY : " , accuracy*100
     #Accuracy on testing data
     correctclassification= []
     for eachline in datadicttest:
        correctclassification.append(eachline["class"])
     ourclassification = classifyNB(datadicttest, priorprob , P, allclasses,var2,mu2)
     accuracy = checkaccuracy(correctclassification, ourclassification)
     print "TESTING ACCURACY : " , accuracy*100


     ######### KNN ##########
     ######## MODELLING #######
     print "\n \n#### KNN Classification ####"
     #get all features
     allfeatures=[]
     for eachfeature, value in datadictraining[0].items():
         allfeatures.append(eachfeature)

     # Find mean per class
     muperfeature = muperfeaturecalc(datadictraining,eachfeature)
     # Find variance and Standarddeviation
     sigmaperfeature = varsdperfeaturecalc(datadictraining,muperfeature,eachfeature)

     ###### CLASSIFY #########
     #Accuracy on training data
     k = [1,3,5,7]
     #Accuracy on testing data
     correctclassification= []
     for eachline in datadictraining:
        correctclassification.append(eachline["class"])
     print "TRAINING ACCURACY"
     train = 1
     for eachk in k:
         ourclassificationL1,ourclassificationL2 = classifyKNN(datadictraining,datadictraining, muperfeature , sigmaperfeature,eachk, train)
         accuracyL1 = checkaccuracy(correctclassification, ourclassificationL1)
         accuracyL2 = checkaccuracy(correctclassification, ourclassificationL2)
         print "L1, K = " ,eachk ," : ", accuracyL1*100
         print "L2, K = " ,eachk ," : ", accuracyL2*100


     print "TESTING ACCURACY  "
     correctclassification=[]
     train = 0
     for eachline in datadicttest:
        correctclassification.append(eachline["class"])
     for eachk in k:
         ourclassificationL1,ourclassificationL2 = classifyKNN(datadictraining,datadicttest, muperfeature , sigmaperfeature,eachk, train)
         accuracyL1 = checkaccuracy(correctclassification, ourclassificationL1)
         accuracyL2 = checkaccuracy(correctclassification, ourclassificationL2)
         print "L1, K = " ,eachk ," : ", accuracyL1*100
         print "L2, K = " ,eachk ," : ", accuracyL2*100