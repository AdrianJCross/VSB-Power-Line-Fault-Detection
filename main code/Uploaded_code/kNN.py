#kNN

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split
import sys
import load



def main():
    Xtrain, ytrain = load.load_training('training_data_new.csv')
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xtrain, ytrain, test_size=0.20)
    nXtrain, nXtest = load.norm(Xtrain, Xtest)
    pXtrain, pXtest = load.pca(nXtrain, nXtest, 0.1)
    fXtrain, fXtest = load.fld(Xtrain, ytrain, Xtest)
    fXtest=fXtest.reshape(-1,1)
    fXtrain=fXtrain.reshape(-1,1)
    
    n_conf_matrix_arr=[]
    p_conf_matrix_arr=[]
    f_conf_matrix_arr=[]
    n_accuracy_arr=[]
    p_accuracy_arr=[]
    f_accuracy_arr=[]
    k_lab=[]
    for k in range(1,10):
        print('Testing for k=',k)
        conf_matrix,accuracy=load.cross_val(nXtrain,nXtest,ytrain,ytest,'kNN',k) 
        k_lab.append(k)  
        n_conf_matrix_arr.append(conf_matrix)
        n_accuracy_arr.append(accuracy)
        
        conf_matrix,accuracy=load.cross_val(pXtrain,pXtest,ytrain,ytest,'kNN',k) 
        p_conf_matrix_arr.append(conf_matrix)
        p_accuracy_arr.append(accuracy)
        
        conf_matrix,accuracy=load.cross_val(fXtrain,fXtest,ytrain,ytest,'kNN',k) 
        f_conf_matrix_arr.append(conf_matrix)
        f_accuracy_arr.append(accuracy)
        

    plt.figure() #plots accuracy vs k value
    plt.plot(k_lab,n_accuracy_arr,label='nX')
    plt.plot(k_lab,p_accuracy_arr,label='pX')
    plt.plot(k_lab,f_accuracy_arr,label='fX')
    plt.xlabel('k value')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('k_var_acc.png')
    plt.show()
        
if __name__ == "__main__":
    main()
    
    
    
    

    
