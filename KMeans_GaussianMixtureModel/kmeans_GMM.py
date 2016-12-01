import json
import math
import collections
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import matplotlib.lines as mlines

def parsefile():
     filename1 = 'hw5_blob.csv'
     filename2 = 'hw5_circle.csv'
     with open(filename1) as fp1 , open(filename2) as fp2:
         blob = np.loadtxt(fp1, delimiter=",")
         circle = np.loadtxt(fp2, delimiter=",")
     return blob, circle

def getdistance( xn, uk):
    return np.linalg.norm(xn-uk)
   # return sum((xn-uk)**2)

def KMeans(eachdata, k):
    centersrandom = np.random.randint(len(eachdata), size=k)
    centroid = eachdata[centersrandom]          ## initialize uk to some values, k centroid values
    clusterassign = [-1]*len(eachdata)
    tocontinue = True

    while tocontinue == True:
        tocontinue = False
        thisclusterk=[-1]*len(eachdata)
        for n in range( 0  , len(eachdata)):
            axn = np.asarray([getdistance( eachdata[n], centroid[j]) for j in range(0,k)])
            thisclusterk[n] = axn.argmin()    # Step 2 , mininmize J

            #clusterassign[n]=thisclusterk[n]
        for n in range(0,len(eachdata)):
            if  clusterassign[n]!= thisclusterk[n]:   ## if cluster assignments are not changed
                tocontinue = True

        ## update the centroids
        for eachk in range(0,k):
            sumofthisclusterx = 0.0
            sumofthisclustery = 0.0
            numk =0
            for n in range( 0  , len(eachdata)):
                if thisclusterk[n] == eachk:
                    sumofthisclusterx = sumofthisclusterx + eachdata[n][0]
                    sumofthisclustery= sumofthisclustery + eachdata[n][1]
                    numk=numk+1
            if(numk==0):  numk=1
            centroid[eachk] = [sumofthisclusterx/numk , sumofthisclustery/numk]
        #keep the clusters
        for n in range(0,len(eachdata)):
            clusterassign[n]=thisclusterk[n]
    return clusterassign

def kernelfunc(xn, yn):
    sigma=0.1
    #return math.exp(-gamma*(sum((xn-yn)**2)))
    #return  math.exp((0-math.pow(-np.linalg.norm(xn-yn), 2))/(2.*sigma**2))
    #return math.exp(0-gamma*(np.linalg.norm(xn-yn)))
    return math.sqrt((sum(xn**2))*(sum(yn**2)))
    #return math.sqrt(((xn[0]*xn[0]) + (xn[1]*xn[1]))  *((yn[0]*yn[0]) + (yn[1]*yn[1])))
    #return math.sqrt(( ((xn[1] - xn[0])*(xn[1] - xn[0])) + ((yn[1]-yn[0])*(yn[1]-yn[0])) ) /  (np.asarray([xn,yn])).var() )
    #c = 0
    #d=1
    #return (xn.T.dot(yn)+c)**2.0


def getkerneldistance( xn, n,  eachdata,j,clusterassign,NK):
    datafunc = []
    for i in range(0,len(eachdata)):
        temp=0.0
        if(clusterassign[i]==j ):
            temp = ( (kernelfunc(eachdata[i],eachdata[i])) / (NK*NK))- ((2*kernelfunc(xn,eachdata[i]))/ NK )
        datafunc.append( kernelfunc(xn,xn)  + temp)
    return sum(datafunc)

def getkerneldistanceinit( xn, n,  eachdata, centroid,j):
    datafunc = []
    temp = ( (kernelfunc(centroid[j],centroid[j])) )- ((2*kernelfunc(xn,centroid[j])))
    datafunc.append( kernelfunc(xn,xn)  + temp)
    return sum(datafunc)

def  clusterassignmentkernelinit(centroid,eachdata,k):
    clusterassign = [-1]*len(eachdata)
    for n in range( 0  , len(eachdata)):
        axn = np.asarray([getkerneldistanceinit( eachdata[n], n ,eachdata, centroid,j) for j in range(0,k)])
        clusterassign[n]= axn.argmin()
    return clusterassign


def KernelKMeans(eachdata, k):
    centersrandom = np.random.randint(len(eachdata), size=k)
    centroid = eachdata[centersrandom]          ## initialize uk to some values, k centroid values
    clusterassign = clusterassignmentkernelinit(centroid,eachdata,k)
    tocontinue = True

    while tocontinue == True:
        tocontinue = False
        nk=[-1]*k
        for eachki in range(0,k):
            temp =0.0
            for i in range( 0  , len(eachdata)):
                if( clusterassign[i]==k): temp=temp+1
            if temp==0.0 : temp = 1
            nk[eachki] = temp
        thisclusterk=[-1]*len(eachdata)
        for n in range( 0  , len(eachdata)):
            axn = np.asarray([getkerneldistance( eachdata[n], n ,eachdata,j,clusterassign,nk[j]) for j in range(0,k)])
            thisclusterk[n] = axn.argmin()    # Step 2 , mininmize J

        ## if cluster assignments are not changed
        for n in range(0,len(eachdata)):
            if  clusterassign[n]!= thisclusterk[n]:
                tocontinue = True

        for n in range(0,len(eachdata)):
            clusterassign[n]=thisclusterk[n]
    return clusterassign

def gaussianfunction( mu , cov,x):
    aa = 0- ((x-mu).T).dot(np.linalg.inv(cov)).dot(x-mu)
    bb = math.exp((aa/2))
    den = math.sqrt(((2*math.pi)**2)* np.linalg.det(cov))
    return bb/den

def calcloglikelihood(eachdata, mu, pi, cov):
    result=0.0
    for k in range (0,len(eachdata)):
        temp1 = sum ( [( pi[i] * gaussianfunction(mu[i],cov[i],eachdata[k])) for i in range(0,3)])
        temp = math.log (temp1)
        result = result + temp
    return result

def calcestep(eachdata,mu,pi,cov):
    ric=[0]*len(eachdata)
    for n in range(len(eachdata)):
        numerator =[]
        for i in range(  0 , len(mu)):
            numerator.append( pi[i]*gaussianfunction(mu[i],cov[i],eachdata[n]))
        ric[n]=np.array(numerator)/sum(numerator)
    return np.asarray(ric)

def calcMStep(eachdata,mu,pi,cov,ric):
    pinew=ric.sum(axis=0)/ric.sum()
    munew = []
    covnew = []
    for k in range(0,len(ric[0])):
        munew.append(eachdata*(ric[:,k:k+1]).sum(axis=0)/ric[:,k].sum(axis=0) )
    for k in range(0,len(ric[0])):
        covnew.append((ric[:,k:k+1]*(eachdata-mu[k])).T.dot(eachdata-mu[k]) / ric[:,k].sum(axis=0))
    return munew, pinew, covnew

def initestep(eachdata, k):
    #r=KMeans(eachdata,k)
    shuffledpoints = [[],[],[]]
    newpoints = list(eachdata)
    np.random.shuffle(newpoints)
    shuffledpoints[0] = newpoints[0:200]
    shuffledpoints[1] = newpoints[200: 400 ]
    shuffledpoints[2] = newpoints[400:500]
    mu = [np.array(shuffledpoints[0]).mean(axis=0),np.array(shuffledpoints[1]).mean(axis=0),np.array(shuffledpoints[2]).mean(axis=0)]
    cov = [np.cov(np.array(shuffledpoints[0]).T),np.cov(np.array(shuffledpoints[1]).T),np.cov(np.array(shuffledpoints[2]).T)]
    pi = [len(shuffledpoints[0])/600.0,len(shuffledpoints[1])/600.0,len(shuffledpoints[2])/600.0]
    return mu,pi,cov

def Gaussianmixturemodel (eachdata,k):
    index = 0
    #fig = plt.figure()

    maxloglikelihood = []
    bestrunindextemp= 0
    tempmax = -sys.maxint
    for i in range ( 0 , 5):
        thisrun =0
        thisloglikelihood = []
        mu,pi,cov = initestep(eachdata, k)
        lastloglikelihoodtemp= 0.0

        for iteration in range(0,1000):
            ric = calcestep(eachdata,mu,pi,cov)
            mu,pi,cov = calcMStep(eachdata,mu,pi,cov,ric)
            thisloglikelihoodtemp = calcloglikelihood(eachdata, mu, pi, cov)
            thisloglikelihood.append(thisloglikelihoodtemp)
            #if(abs(thisloglikelihoodtemp- lastloglikelihoodtemp ) <0.0001):
            # get max of this run
            if (thisloglikelihoodtemp > tempmax):
                thisrun = 1
                tempmax = thisloglikelihoodtemp
                maxloglikelihood = []
                musaved,pisaved,covsaved = mu,pi,cov

            if( (abs(thisloglikelihoodtemp- lastloglikelihoodtemp ) <0.0000000001)):
                break
            lastloglikelihoodtemp = thisloglikelihoodtemp
        if thisrun==1:
            maxloglikelihood.append(thisloglikelihood)
            bestrunindextemp= i

        plt.ylabel("Log Likelihood")
        plt.xlabel("Iterations")
        colorstouse = ['y' ,'g','r','b','c']
        for i in range(3):
                plt.plot( thisloglikelihood,colorstouse[index])

        #ax = plt.subplot(2,3,index+1)

        #plt.plot( thisloglikelihood, '-o')
        index = index+1
    bestrunindex = [""]*5
    #print  bestrunindextemp
    bestrunindex[bestrunindextemp]= "Best Run"
    run1 = mlines.Line2D([], [], color='y', marker='*', label='1st Run'+bestrunindex[0])
    run2 = mlines.Line2D([], [], color='g', marker='*', label='2nd Run'+bestrunindex[1])
    run3 = mlines.Line2D([], [], color='r', marker='*', label='3rd Run'+bestrunindex[2])
    run4 = mlines.Line2D([], [], color='b', marker='*', label='4th Run'+bestrunindex[3])
    run5 = mlines.Line2D([], [], color='c', marker='*', label='5th Run'+bestrunindex[4])
    lines = [run1, run2,run3,run4,run5]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels)

    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels)
    plt.show()

    maxassignment = ric.argmax(axis=1)
    print "GAUSSIAN MIXTURE MODEL "

    print "                     Mean                             Covariance"
    print " CLUSTER 1   " + str(musaved[0].tolist()) + "  " + str(covsaved[0].tolist())
    print " CLUSTER 2   " + str(musaved[1].tolist()) + "  " + str(covsaved[1].tolist())
    print " CLUSTER 3   " + str(musaved[2].tolist()) + "  " + str(covsaved[2].tolist())
    return  maxassignment

if __name__ == "__main__":
     blob,circle = parsefile()

     ## K MEANS ##
     # do we have to normalize the data?

     fig = plt.figure()
     index = 0

     colorstouse = ['y' ,'g','r','b','c']
     for eachdata in [blob,circle]:
         for eachk in [2,3,5]:
             clusterassignment = KMeans (eachdata,eachk)
             colorsindex = [colorstouse[k] for k in clusterassignment]
             ax = plt.subplot(2,3,index+1)
             plt.ylabel("y values")
             plt.xlabel("x values")
             plt.title("K = "+ str(eachk))
             ax.scatter(eachdata[:,0], eachdata[:,1],color=colorsindex)
             index = index+1
     plt.show()

     ## Kernel K MEANS ##
     # do we have to normalize the data?
     index = 0
     for eachdata in [circle]:
         eachk = 2
         clusterassignment = KernelKMeans (eachdata,eachk)
         colorsindex = [colorstouse[k] for k in clusterassignment]
         ax = plt.subplot(1,1,index+1)
         plt.ylabel("y values")
         plt.xlabel("x values")
         plt.title("K = "+ str(eachk))
         ax.scatter(eachdata[:,0], eachdata[:,1],color=colorsindex)
         index = index+1
     plt.show()


     ## Gaussian Mixture Model
     index = 0
     for eachdata in [blob]:
         eachk = 3
         maxassignment = Gaussianmixturemodel (eachdata,eachk)
         colorsindex = [colorstouse[k] for k in maxassignment]
         ax = plt.subplot(1,1,index+1)
         plt.ylabel("y values")
         plt.xlabel("x values")
         plt.title("K = "+ str(eachk))
         ax.scatter(eachdata[:,0], eachdata[:,1],color=colorsindex)
         plt.show()

