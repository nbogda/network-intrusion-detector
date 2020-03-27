'''
The following is from the Jantz man himself:

"Suggestions for “Helloworld”: Deploy your data set on the Neuro-
Firewall cluster. Train and test a single (simple) classifier, such as k-
means clustering."

And what Jantz says, goes.
'''

import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics
import sys
import seaborn as sn

def split_train_test(data, pca=False):
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

    # pca is optional
    desc = ""
    if pca:
        X = pca_data(X)
        desc = "_PCA"

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    np.save("X_train%s.npy" % desc, X_train)
    np.save("X_test%s.npy" % desc, X_test)
    np.save("y_train%s.npy" % desc, y_train)
    np.save("y_test%s.npy" % desc, y_test)
    return X_train, X_test, y_train, y_test

def pca_data(X):
    pca = PCA()
    X = pca.fit_transform(X)

    #get variance explained
    explained_variance = pca.explained_variance_ratio_

    #make first plot of just principal components
    fig1 = plt.figure()
    plt.plot(explained_variance)
    plt.title("Principal Components")
    plt.ylabel("Percent of Variance Explained")
    plt.xlabel("Principal Component")
    plt.savefig("graphs/principal_comp_.png")

    #select what percent var to keep
    desired_var = 0.9  # try out different values for this, make graph
    #select how many eigenvalues to keep
    cumsum = np.cumsum(explained_variance)
    k = np.argwhere(cumsum > desired_var)[0]

    #make second plot of cum var explained
    fig2 = plt.figure()
    plt.plot(cumsum)
    plt.title("Variance Explained")
    plt.plot(k, cumsum[k], 'ro', label="Eigenvalue #%d with %.2f Variance" % (k, desired_var))
    plt.legend()
    plt.ylabel("Cumulative Percent of Variance Explained")
    plt.xlabel("Principal Component")
    plt.savefig("graphs/var_exp_.png")

    pca = PCA(n_components=int(k))
    X = pca.fit_transform(X)
    return X
            
# prefer to use the F1 score with weighted average to account for label imbalance
# F1 score is between 0 and 1. The closer to 1, the better.
def choose_k(X_train, y_train, X_test, y_test):
    max_f1_score = -1
    best_knn = None
    f1_scores = []

    k_range = np.arange(1, 15)
    for k in k_range:
        print("On k=%d" % k)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1_score)
        if f1_score > max_f1_score:
            max_f1_score = f1_score
            best_knn = knn

    np.save("f1_scores.npy", f1_scores)

    # plottin' our k's
    fig = plt.figure(figsize=((10, 5)))
    plt.plot(k_range, f1_scores, marker='o')
    plt.title("kNN - F1 Score vs k")
    plt.xlabel("k")
    plt.ylabel("F1 Score")
    plt.savefig("graphs/kNN_k_selection.png")

    return f1_scores.index(max(f1_scores)) + 1, max(f1_scores), best_knn


# making a fancy confusion matrix
def show_results_with_best_k(k, f1, X_test, y_test, knn):
    
    labels = list(set(y_test))
    y_pred = knn.predict(X_test)

    # get confusion matrix and normalize it
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(17, 15))
    sn.heatmap(cm, annot=True, ax=ax, fmt='.3f', cmap='Greens', linewidths=0.25, linecolor='black')
    ax.set_xlabel('Predicted Labels', fontsize='large')
    ax.set_ylabel('True Labels', fontsize='large')
    ax.set_title("Confusion matrix for kNN k=%d, F1 Score = %.4f" % (k, f1), fontsize='large')
    ax.xaxis.set_ticklabels(labels, rotation='vertical', fontsize='large')
    ax.yaxis.set_ticklabels(labels, rotation='horizontal', fontsize='large')
    plt.savefig("graphs/knn_confusion_matrix.png")

def main():
    # assume that if x_train doesn't exist, then none of the sets exist
    pca = True
    if (not os.path.exists('x_train.npy') and not pca) or (not os.path.exists("X_train_PCA.npy") and pca):
        data = pd.read_csv('../data/train_nodup.csv')
        X_train, X_test, y_train, y_test = split_train_test(data, pca=pca)
        print("Transformed data")
    else:
        desc = ""
        if pca:
            desc = "_PCA"
        X_train = np.load("X_train%s.npy" % desc)
        X_test = np.load("X_test%s.npy" % desc)
        y_train = np.load("y_train%s.npy" % desc)
        y_test = np.load("y_test%s.npy" % desc)
        print("Loaded in data")

    # returns best k and its f1 score
    # k, f1, knn = choose_k(X_train, y_train, X_test, y_test)
   
    
    k = 1
    f1 = 0.9995
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    show_results_with_best_k(k, f1, X_test, y_test, knn)


if __name__ == "__main__":
    main()

'''
Plans for "Hello World":
    1. Split the above data into train and test (70% / 30%)
        1.a. Do we need to normalize the data?
    2. Fit to kmeans cluster
    3. Perform k-fold cross validation (either 3 or 5 folds, depends on how long it takes to run)
    4. Get performance metrics, such as:
        a. Accuracy
        b. F1 Score
        c. Confusion Matrix
        d. ROC curve 
        e. Cluster graph
'''


