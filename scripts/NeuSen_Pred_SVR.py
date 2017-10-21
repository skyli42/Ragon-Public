#!/usr/bin/python
#
#
import warnings
warnings.filterwarnings("ignore")

from os import listdir
from os.path import isfile, join
import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import os, pickle, networkx,sys,re
from optparse import OptionParser
from Bio import motifs,SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
import xlrd
import seaborn as sns
import pandas as pd
import operator
import math,string
import random
import statsmodels.api
import shannon.continuous as continuous
import shannon.discrete as discrete
from ete3 import Tree

from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.svm import SVR,NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV,ShuffleSplit,train_test_split,KFold
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, RandomizedLasso, ElasticNet, ElasticNetCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import r2_score, mean_squared_error
sns.set_style(style='white')

#customized module
import sys
sys.path.append('C:/Users/sky/Documents/Ragon/scripts') #add module path to system


#preProcess=0
progdir=os.path.dirname(sys.argv[0])
usage = "usage: %prog [options] <AbModel>"
parser = OptionParser(usage=usage)
(opts, args) = parser.parse_args()
# if len(args) != 3:  
    # parser.error("incorrect number of arguments")
AbModel="PGT121"

#AbModel='3BNC117'#PGT121, VRC01, 3BNC117
ProjectPath='./'
TrainData='../data/'+AbModel+'_Neu_OccAA.csv'

#=======================================================================#
#Functions
#=======================================================================#
def GetSVRmodel(task,AllTable,SavePath,gamma=0.0019953,C=100,degree=3):
    RemainCol=AllTable.columns.tolist()[0:4]
    for col in AllTable.columns.tolist()[4:]: #Removing the features with no-changed profiles
        if len(list(set(AllTable.loc[:,col].tolist())))>1:
            RemainCol.append(col)
    AllTable=AllTable.loc[:,RemainCol]#Final working feamatrix
    
    FeaMatrix=numpy.array(AllTable.iloc[:,4:])#All features
    FeaName=AllTable.iloc[:,4:].columns.tolist()
    yMatrix=numpy.log2(numpy.array(AllTable.IC50_mean)) #Take log2 of neutralization
    log2yMean=numpy.mean(yMatrix)
    log2ystd=numpy.std(yMatrix)
    yMatrix=(yMatrix-log2yMean)/log2ystd #Zscore
    ID_index=AllTable.index.tolist()
    

    if 1 in task: #SVR parameter search
        C_range = numpy.logspace(-1, 8, 10)#C_range = numpy.logspace(-2, 12, 10) #DH270.5
        gamma_range = numpy.logspace(-9, 0, 11)#gamma_range = numpy.logspace(-12, 0, 11)

        #degree_range =numpy.array([3,4,5])#
        param_grid = dict(gamma=gamma_range, C=C_range)
        grid = GridSearchCV(SVR(kernel='rbf',cache_size=1000), param_grid=param_grid,verbose = 1, n_jobs = -1, cv=5,scoring='neg_mean_squared_error')
        grid.fit(FeaMatrix, yMatrix)
        gamma=grid.best_params_['gamma']
        C=grid.best_params_['C']
        #degree=grid.best_params_['degree']
        print("SVR ::: The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_), flush = True)
        plot=1
        if plot:
            scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),len(gamma_range))
            plt.figure(figsize=(8, 6))
            plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
            plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,norm=MidpointNormalize(vmin=numpy.min(scores), midpoint=numpy.max(scores)))
            plt.xlabel('gamma')
            plt.ylabel('C')
            plt.colorbar()
            plt.xticks(numpy.arange(len(gamma_range)), gamma_range, rotation=45)
            plt.yticks(numpy.arange(len(C_range)), C_range)
            plt.title('Validation accuracy')
            plt.savefig(SavePath+'SVR_parameterScan.png')   
            #plt.show()

    if 2 in task: #Feature selection
        #Determine Lasso alpha by least angle regression with BIC/AIC criterion or lassoCV
        for ai in [1]:#aic/bic/lassocv show similar performance , lassoLarscv is the worst
            if ai==1:
                model = LassoLarsIC(criterion='aic')
                print('aic', flush = True)
            elif ai==2:
                model = LassoLarsIC(criterion='bic')
                print('bic', flush = True)
            elif ai==3:
                model = LassoLarsCV(cv=10)
                print('lassoLarscv', flush = True)
            else:
                model = LassoCV(cv=10)
                print('lassocv', flush = True)

            model.fit(FeaMatrix, yMatrix)
            alpha_sel = model.alpha_
            print('alpha:'+str(alpha_sel), flush = True)

            l1_ratio_screen=numpy.linspace(0.05,1,100)
            #Select l1_ratio based on ElasticNet regression
            #grid = GridSearchCV(ElasticNet(), param_grid=dict(alpha=[alpha_sel], l1_ratio=l1_ratio_screen), cv=10,scoring='r2')
            #grid.fit(FeaMatrix, yMatrix)
            #print 'Selected l1_ratio:'+str(grid.best_params_['l1_ratio'])+'|Selected alpha:'+str(grid.best_params_['alpha']);
            #Select l1_ratio based on SVR regression
            rs=[]
            for ss in l1_ratio_screen:
                clf = ElasticNet(alpha=alpha_sel, l1_ratio=ss)
                sfm = SelectFromModel(clf) #feature selection based on elasticNetCV
                sfm.fit(FeaMatrix, yMatrix)
                Selected_index=sfm.get_support() #Get true,false index of feature selected
                Selected_FeaMatrix = sfm.transform(FeaMatrix) #Reduce FeaMatrix to the selected features.
                r=Model_CV(Selected_FeaMatrix, yMatrix, 2, SavePath, fileName='ElasticNetFea', iteration=100, kfold=10, gamma=gamma, C=C, plot=0)
                rs.append(r)
            
            l1_ratio=l1_ratio_screen[rs.index(max(rs))]
            print(l1_ratio)
            clf = ElasticNet(alpha=alpha_sel, l1_ratio=l1_ratio)
            sfm = SelectFromModel(clf) #feature selection based on elasticNetCV
            sfm.fit(FeaMatrix, yMatrix)
            Selected_index=sfm.get_support() #Get true,false index of feature selected
            Selected_FeaMatrix = sfm.transform(FeaMatrix) #Reduce FeaMatrix to the selected features.
            #print Selected_FeaMatrix.shape
            for s in [2]:
                print('Selected features ElasticNet regularization', flush = True)
                Model_CV(Selected_FeaMatrix, yMatrix, s, SavePath,fileName='ElasticNetFea',iteration=100,kfold=10,gamma=gamma,C=C,adjMean=log2yMean,adjStd=log2ystd, plot=1, nfold=1)

        #Output the results
        SelFeaTable=pd.DataFrame(data=FeaMatrix[:,Selected_index], index=ID_index,columns=[x for x, y in zip(FeaName, Selected_index) if y])
        SelFeaTable=pd.concat([SelFeaTable,AllTable.IC50_mean], axis=1)
        writer = pd.ExcelWriter(SavePath+'_model_SelectfeaMatrix.xlsx')
        SelFeaTable.to_excel(writer,'Select_FeaMatrix')
        writer.save()
        with open(SavePath+'_PredModel.pckl', 'wb') as f:
            pickle.dump([SelFeaTable,gamma,C], f)
    return 

def Model_CV(FeaMatrix, yMatrix, task, OutPath, iteration=100, kfold=5, gamma=0.0019953, C=100, degree=3, fileName='',plot=1,adjMean=1,adjStd=1,nfold=0):
    svr_rbf = SVR(kernel='rbf', C=C, gamma=gamma,cache_size=1000)#kernel='poly'

    if task==1: #Resampling
        rhos=[];aucs=[]
        rs = ShuffleSplit(n_splits=iteration, test_size=0.2, random_state=42)
        for train_index, test_index in rs.split(FeaMatrix):
            X_train, X_test = FeaMatrix[train_index], FeaMatrix[test_index]
            y_train, y_test = yMatrix[train_index], yMatrix[test_index]
            y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)#Unbalanced distributed glycan occupancy, using sample weight to correct that
            rho, pval = scipy.stats.spearmanr(y_test,y_rbf)
            rhos.append(rho)
        print("SVR|Resampling %s r= %0.4f" % (iteration,numpy.mean(numpy.array(rhos))), flush = True)
    elif task==2: #K-fold cross validation
        kf = KFold(n_splits=kfold)
        rhos=[];aucs=[];y_data=[];y_pred=[];fprs=[];tprs=[];
        for train_index, test_index in kf.split(FeaMatrix):
            X_train, X_test = FeaMatrix[train_index], FeaMatrix[test_index]
            y_train, y_test = yMatrix[train_index], yMatrix[test_index]
            y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
            rho, pval = scipy.stats.spearmanr(y_test,y_rbf)
            rhos.append(rho)

            y_rbf=(y_rbf*adjStd)+adjMean#Transform back
            y_test=(y_test*adjStd)+adjMean#Transform back
            y_data.append(list(y_test))
            y_pred.append(list(y_rbf))
        
        if plot==1:
            print("SVR|%s-fold r= %0.4f" % (kfold,numpy.mean(numpy.array(rhos))), flush = True)

            # print("r2: %0.4f" %(r2_score(y_test, y_pred)))
            f=plt.figure(figsize=(10, 10))
            gs = gridspec.GridSpec(1, 1)
            gs.update(left=0.08,right=0.95,bottom=0.05,top=1,wspace=0.5,hspace=0.5)
            ax = plt.subplot(gs[0, 0])
            for i in range(0,len(y_data)):
                ax.scatter(y_data[i],y_pred[i], s=20, c='blue', alpha=0.5);
            #ax.set_xlim([0, 1]);ax.set_ylim([0, 1])
            ax.set_xlabel('Log2 observed IC50', fontsize=15);ax.set_ylabel('Log2 predicted IC50', fontsize=15)
            plt.text(0.1, 0.8, 'rho = %.2f' % numpy.mean(numpy.array(rhos)), fontsize=15)
            sns.set_style("white")
            plt.savefig(OutPath+'_Predict_CV_SVR_'+fileName+'.eps') 
            #plt.show()
        if nfold==1: #calculate leave-one-out CV
            y_data=[];y_pred=[];
            for s in range(0,len(yMatrix)):
                test_index=s
                train_index=list(range(0,len(yMatrix)))
                del train_index[s]

                X_train, X_test = FeaMatrix[train_index], FeaMatrix[test_index]
                y_train, y_test = yMatrix[train_index], yMatrix[test_index]
                y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test.reshape(1, -1))
                y_rbf=(y_rbf*adjStd)+adjMean#Transform back
                y_test=(y_test*adjStd)+adjMean#Transform back
                y_data.append(y_test)
                y_pred.append(y_rbf)
            print(numpy.asarray(y_data).shape, flush = True)
            print(numpy.asarray(y_pred).shape, flush = True)
            # print(y_pred)
            rho1, pval1 = scipy.stats.spearmanr(y_data,y_pred)
            
            print(mean_squared_error(y_data, y_pred).ravel())
            print("SVR|leave-one-out r= %0.4f p-value= %0.4f Rsquare=%0.4f" % (rho1,pval1,rho1*rho1), flush = True)

    return numpy.mean(numpy.array(rhos))

#=======================================================================#
#Class
#=======================================================================#
class MidpointNormalize(Normalize):
    # Utility function to move the midpoint of a colormap to be around
    # the values of interest.
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return numpy.ma.masked_array(numpy.interp(value, x, y))

#=======================================================================#
#The scrtip starts here
#=======================================================================#
def main():
    Name=re.sub('_Neu_OccAA.csv','',re.sub('Model_v2\/bNAb_prediction\/','',TrainData))
    AllTable=pd.read_csv(ProjectPath+TrainData,sep=',',index_col=0)
    GetSVRmodel([1,2],AllTable,'../output/')

if __name__=='__main__':
    main()

