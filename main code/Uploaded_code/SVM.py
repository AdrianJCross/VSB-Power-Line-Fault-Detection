#SVM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split
import sys
import load
import time
path='/home/across/UTK_PhD/Machine_learning_fall_2019/final_project/'


def main():
    Xtrain, ytrain = load.load_training('training_data_new.csv')
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xtrain, ytrain, test_size=0.20)
    nXtrain, nXtest = load.norm(Xtrain, Xtest)
    pXtrain, pXtest = load.pca(nXtrain, nXtest, 0.1)
    fXtrain, fXtest = load.fld(Xtrain, ytrain, Xtest)
    fXtest=fXtest.reshape(-1,1)
    fXtrain=fXtrain.reshape(-1,1)
    print('AAAA')
    start = time.clock()
    cn1,an1=load.cross_val(nXtrain,nXtest,ytrain,ytest,'SVM',1,kern='linear')
    end=time.clock()
    print('Time for normalized data is ',end-start)
    
    #sensitivity calculation TP/(TP+FN)
    
    
    cn2,an2=load.cross_val(nXtrain,nXtest,ytrain,ytest,'SVM',2,kern='poly') 
    cn3,an3=load.cross_val(nXtrain,nXtest,ytrain,ytest,'SVM',3,kern='poly') 
    cn4,an4=load.cross_val(nXtrain,nXtest,ytrain,ytest,'SVM',1,kern='rbf') 
    cn5,an5=load.cross_val(nXtrain,nXtest,ytrain,ytest,'SVM',1,kern='sigmoid')
    print('Completed for normalized data')
    start = time.clock()
    cp1,ap1=load.cross_val(pXtrain,pXtest,ytrain,ytest,'SVM',1,kern='linear') 
    end=time.clock()
    print('Time for PCA data is ',end-start)
    cp2,ap2=load.cross_val(pXtrain,pXtest,ytrain,ytest,'SVM',2,kern='poly') 
    cp3,ap3=load.cross_val(pXtrain,pXtest,ytrain,ytest,'SVM',3,kern='poly') 
    cp4,ap4=load.cross_val(pXtrain,pXtest,ytrain,ytest,'SVM',1,kern='rbf') 
    cp5,ap5=load.cross_val(pXtrain,pXtest,ytrain,ytest,'SVM',1,kern='sigmoid')
    print('Completed for PCA data')
    start = time.clock()
    cf1,af1=load.cross_val(fXtrain,fXtest,ytrain,ytest,'SVM',1,kern='linear') 
    end=time.clock()
    print('Time for FLD data is ',end-start)
    #cf2,af2=load.cross_val(fXtrain,fXtest,ytrain,ytest,'SVM',2,kern='poly')
    print('1')
    #cf3,af3=load.cross_val(fXtrain,fXtest,ytrain,ytest,'SVM',3,kern='poly') 
    print('2')
    #cf4,af4=load.cross_val(fXtrain,fXtest,ytrain,ytest,'SVM',1,kern='rbf')
    print('3')
    #cf5,af5=load.cross_val(fXtrain,fXtest,ytrain,ytest,'SVM',1,kern='sigmoid')
    print('Completed for FLD data')
    n_groups = 5
    af2=af3=af4=af5=0
    nX_data = [an1,an2,an3,an4,an5]
    pX_data = [ap1,ap2,ap3,ap4,ap5]
    fX_data = [af1,af2,af3,af4,af5]
    
    #sensitivity calculation TP/(TP+FN)
    sn1 = 100*cn1[1,1]/(cn1[1,1]+cn1[1,0])
    sn2 = 100*cn2[1,1]/(cn2[1,1]+cn2[1,0])
    sn3 = 100*cn3[1,1]/(cn3[1,1]+cn3[1,0])
    sn4 = 100*cn4[1,1]/(cn4[1,1]+cn4[1,0])
    sn5 = 100*cn5[1,1]/(cn5[1,1]+cn5[1,0])
    
    sn1 = 0
    sn2 = 0
    print(cn1)
    sp1 = 100*cp1[1,1]/(cp1[1,1]+cp1[1,0])
    sp2 = 100*cp2[1,1]/(cp2[1,1]+cp2[1,0])
    sp3 = 100*cp3[1,1]/(cp3[1,1]+cp3[1,0])
    sp4 = 100*cp4[1,1]/(cp4[1,1]+cp4[1,0])
    sp5 = 100*cp5[1,1]/(cp5[1,1]+cp5[1,0])
    
    sp1 = 0
    sp2 = 0
    
    
    print(sp3)
    print(cp3)
    sf1 = 100*cf1[1,1]/(cf1[1,1]+cf1[1,0])
    sf1 = 0
    sf2 = 0
    sf3 = 0
    sf4 = 0
    sf5 = 0
    
    X = np.arange(5)
    
    bars=('Linear', '2nd Poly', '3rd Poly', 'RBF', 'Sigmoid')
    y_pos = np.arange(len(bars))
    plt.bar(X + 0.00, nX_data, color = 'b', width = 0.25,label='nX')
    plt.bar(X + 0.25, pX_data, color = 'g', width = 0.25,label='pX')
    plt.bar(X + 0.50, fX_data, color = 'r', width = 0.25,label='fX') 
    plt.xlabel('SVM boundary')
    plt.xticks(y_pos+0.5, bars, color='black', fontweight='bold', fontsize='8', horizontalalignment='right')
    #plt.xticks('Linear', '2nd Poly', '3rd Poly', 'RBF', 'Sigmoid')
    plt.ylabel('Accuracy')
    plt.ylim(0,105)
    plt.legend()
    plt.savefig(path+'SVM_acc.png')
    plt.ylim(90,100)
    plt.savefig(path+'SVM_acc_zoom.png')
    plt.clf()
    
    nX_data = [sn1,sn2,sn3,sn4,sn5]
    pX_data = [sp1,sp2,sp3,sp4,sp5]
    fX_data = [sf1,sf2,sf3,sf4,sf5]
    bars=('Linear', '2nd Poly', '3rd Poly', 'RBF', 'Sigmoid')
    y_pos = np.arange(len(bars))
    plt.bar(X + 0.00, nX_data, color = 'b', width = 0.25,label='nX')
    plt.bar(X + 0.25, pX_data, color = 'g', width = 0.25,label='pX')
    plt.bar(X + 0.50, fX_data, color = 'r', width = 0.25,label='fX') 
    plt.xlabel('SVM boundary')
    plt.xticks(y_pos+0.5, bars, color='black', fontweight='bold', fontsize='8', horizontalalignment='right')
    #plt.xticks('Linear', '2nd Poly', '3rd Poly', 'RBF', 'Sigmoid')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.savefig(path+'SVM_sens.png')
    plt.ylim(95,100)
    plt.savefig(path+'SVM_sens_zoom.png')
    
if __name__ == "__main__":
    main()

    
