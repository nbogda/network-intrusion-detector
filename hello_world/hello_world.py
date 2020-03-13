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

data = pd.read_csv('../data/train_nodup.csv')

'''
Plans for "Hello World":
    1. Split the above data into train and test (70% / 30%)
    2. Fit to kmeans cluster
    3. Perform k-fold cross validation (either 3 or 5 folds, depends on how long it takes to run)
    4. Get performance metrics, such as:
        a. Accuracy
        b. F1 Score
        c. Confusion Matrix
        d. ROC curve 
        e. Cluster graph

'''


