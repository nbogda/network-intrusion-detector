'''
The following is from the Jantz man himself:

"Suggestions for “Helloworld”: Deploy your data set on the Neuro-
Firewall cluster. Train and test a single (simple) classifier, such as k-
means clustering."

And what Jantz says, goes.
'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def split_train_test(data):
    '''
    Will return arrays of these sizes:
        X_train : (752494, 41)
        y_train : (752494, )

        X_test : (322498, 41)
        y_test : (322498, )
    '''

    # convert all but last column to list of lists for data
    X = np.array(data.iloc[:,:-1].values.tolist())
    # convert last column to list for labels
    y = np.array(data.iloc[:,-1].values.tolist())
    return train_test_split(X, y, test_size=0.3)


def main():
    data = pd.read_csv('../data/train_nodup.csv')
    X_train, X_test, y_train, y_test = split_train_test(data)

    # 23 clusters because 23 potential classifications
    kmeans = KMeans(n_clusters=23, random_state=0).fit(X_train)


    # we might need to put a random search here to optimize the parameters of kmeans
    # also I definitely think we could use some PCA here

    


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


