'''
I made this a separate file because we are probably going to use it in other parts of our project
'''

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def read_data(data):
    '''
    Will return arrays of these sizes:
        X_train : (752494, 41)
        y_train : (752494, )

        X_test : (322498, 41)
        y_test : (322498, )
    '''

    # convert all but last column to list of lists for data
    X = np.array(data.iloc[:,:-1].values.tolist())
    X = StandardScaler().fit_transform(X)  # mean of ~0 and variance of 1
    # convert last column to list for labels
    y = np.array(data.iloc[:,-1].values.tolist())
    return X, y

# only call this on training set
def PCA_train_test(X, y):
    # pca is NOT optional
    X, pca = pca_data(X)
    desc = '_PCA'

    np.save("X_train%s.npy" % desc, X)
    np.save("y_train%s.npy" % desc, y)
    return X, y, pca

def pca_data(X):
    pca = PCA()
    X = pca.fit_transform(X)

    #get variance explained
    explained_variance = pca.explained_variance_ratio_
    '''
    #make first plot of just principal components
    fig1 = plt.figure()
    plt.plot(explained_variance)
    plt.title("Principal Components")
    plt.ylabel("Percent of Variance Explained")
    plt.xlabel("Principal Component")
    plt.savefig("graphs/principal_comp_.png")
    '''
    #select what percent var to keep
    desired_var = 0.9
    #select how many eigenvalues to keep
    cumsum = np.cumsum(explained_variance)
    k = np.argwhere(cumsum > desired_var)[0]

    '''
    #make second plot of cum var explained
    fig2 = plt.figure()
    plt.plot(cumsum)
    plt.title("Variance Explained")
    plt.plot(k, cumsum[k], 'ro', label="Eigenvalue #%d with %.2f Variance" % (k, desired_var))
    plt.legend()
    plt.ylabel("Cumulative Percent of Variance Explained")
    plt.xlabel("Principal Component")
    plt.savefig("graphs/var_exp_.png")
    '''
    pca = PCA(n_components=int(k))
    X = pca.fit_transform(X)

    return X, pca
            

def main():

    # training data
    data = pd.read_csv('../../data/train_nodup.csv')
    X, y = read_data(data)
    X, y, pca = PCA_train_test(X, y)
    
    # testing data
    data = pd.read_csv("../../data/test_nodup.csv")
    X, y = read_data(data)
    X_test_pca = pca.fit_transform(X)
    np.save("X_test_PCA.npy", X_test_pca)
    np.save("y_test_PCA.npy", y)
    

if __name__ == "__main__":
    main()


