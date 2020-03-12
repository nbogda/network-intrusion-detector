import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import random as rand
import sys
import collections
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
from itertools import combinations, combinations_with_replacement
from sklearn.metrics import make_scorer, mean_squared_log_error



#Nasty lil dictionary used to determine which node will run which parmiganna combination
jobs = {1:['kNN',0,0], 2:['kNN',0,1], 3:['kNN',1,0], 4:['kNN',1,1], 5:['kNN',2,0], 6:['kNN',2,1],
	7:['MLP',0,0], 8:['MLP',0,1], 9:['MLP',1,0], 10:['MLP',1,1], 11:['MLP',2,0], 12:['MLP',2,1],
	13:['Decision Tree',0,0], 14:['Decision Tree',0,1], 15:['Decision Tree',1,0], 16:['Decision Tree',1,1], 17:['Decision Tree',2,0], 18:['Decision Tree',2,1],
	19:['SVM',0,0], 20:['SVM',0,1], 21:['SVM',1,0], 22:['SVM',1,1], 23:['SVM',2,0], 24:['SVM',2,1],
	25:['Random Forest',0,0], 26:['Random Forest',0,1], 27:['Random Forest',1,0], 28:['Random Forest',1,1], 29:['Random Forest',2,0], 30:['Random Forest',2,1]}


#function to read in the CSV files
def read_CSV(clean_method, preprocessing):
    '''
    clean_method : integer
                   0 - deleted
                   1 - mean
                   2 - to_0
    
    preprocessing : integer
                    0 - ORIGINAL
                    1 - PCA
    '''
    
    file_paths = ["deleted", "mean", "to_0"]
    file_name = ["ORIGINAL", "PCA"]

    #print statement for my sanity
    print("\nYou have selected NaN %s, with %s data\n" % (file_paths[clean_method], file_name[preprocessing]))

    train = pd.read_csv("../../data/NAN_%s/%s_split_data_train.csv" % (file_paths[clean_method], file_name[preprocessing]))
 
    #split data into predictions and predictors
    X = [] #predictors training
    y = [] #predictions training
    
    for index, row in train.iterrows():
        y.append(list(row.iloc[1:13]))
        X.append(list(row.iloc[13:]))
        
    return X, y

def get_params(algorithm):
    '''
    algorithm : name of the algorithm to get params for

    returns dict of params
    '''
    if algorithm == "kNN":
        return { 'n_neighbors' : np.arange(1, 100, 5),
                 'p' : [1, 2, 3] } #different orders of minkowski distance. 1=manhattan, 2=euclidean
    elif algorithm == "MLP": 
        hidden_layers = MLP_structure()
        return { 'hidden_layer_sizes' : hidden_layers,
                 'alpha' : [0.01, 1, 5, 10],
                 'learning_rate_init' : [0.001, 0.01, 0.1, 1, 5],
                 'batch_size' : [1, 10, 30, 200],
                 'activation' : ['logistic', 'relu', 'tanh']}
    elif algorithm == "Decision Tree":
        return { 'criterion' : ["mse", "friedman_mse", "mae"],
                 'min_samples_split' : [2, 4, 6, 8],
                 'min_samples_leaf' : [1, 2, 3, 4],
                 'max_features' : ["auto", "sqrt", "log2"] }
    elif algorithm == "SVM":
        return { 'estimator__kernel' : ['rbf', 'sigmoid', 'poly'],
                 'estimator__gamma' : ['scale', 'auto'],
                 'estimator__C' : [0.1, 1, 5, 10],
                 'estimator__epsilon' : [0.1, 1, 5, 10] }
    elif algorithm == "Random Forest":
        return { 'n_estimators' : [10, 50, 100, 200, 500],
                 'criterion' : ["mse", "friedman_mse", "mae"],
                 'min_samples_split' : [2, 4, 6, 8],
                 'min_samples_leaf' : [1, 2, 3, 4],
                 'max_features' : ["auto", "sqrt", "log2"] }

#this is just to handle exceptions
def custom_scorer(y_true, y_pred):
    score = np.nan
    try:
        #keep this identical to original scoring method from sklearn
        score = mean_squared_log_error(y_true, y_pred) * -1
    except Exception:
        pass
    return score

#for hidden layer/hidden neuron combos in MLP
def MLP_structure():

    hidden_layers = [1, 3, 5]
    hidden_neurons = [5, 10, 20]
    structure = []

    for layer in hidden_layers:
        neuron_layer = list(combinations_with_replacement(hidden_neurons, layer))
        structure += tuple(neuron_layer)
    return structure

def random_search_(algorithm, params, X, y, cm, pp, iters=20, jobs=5):
    '''
    Testing the following algs: 

        kNN, BPNN/MLP, Decision Tree, Random Forest, SVM
    '''
    clf = None
    if algorithm == "kNN":
        clf = KNeighborsRegressor()
    elif algorithm == "MLP":
        #closest to what we did in class
        clf = MLPRegressor(solver="sgd")
    elif algorithm == "Decision Tree":
        clf = DecisionTreeRegressor()
    elif algorithm == "SVM":
        clf = SVR()
        clf = MultiOutputRegressor(clf)
    elif algorithm == "Random Forest":
        clf = RandomForestRegressor()

    custom_neg_MSLE = make_scorer(custom_scorer)
    random_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=iters, n_jobs=jobs, 
                                       scoring=custom_neg_MSLE, refit=True, verbose=2,cv=10)
    random_search.fit(X, y)
    #report(random_search.cv_results_)

    file_paths = ["deleted", "mean", "to_0"]
    file_name = ["ORIGINAL", "PCA"]
    
    #save the model
    best_estimator = random_search.best_estimator_
    joblib.dump(best_estimator, "saved_models/best_%s_%s_%s.joblib" % (algorithm, file_name[pp], file_paths[cm]))

    #write info about the model
    info = pd.read_csv("saved_models/Random_Search_Info.csv", index_col=0)
    best_params = random_search.best_params_
    fit_time = random_search.refit_time_ 
    best_score = np.sqrt(np.abs(random_search.best_score_))
    info.loc["Best %s %s %s" % (algorithm, file_name[pp], file_paths[cm]), "Best Params"] = str(best_params)
    info.loc["Best %s %s %s" % (algorithm, file_name[pp], file_paths[cm]), "Mean RMSLE"] = "%.4f" % best_score
    info.loc["Best %s %s %s" % (algorithm, file_name[pp], file_paths[cm]), "Refit Time"] = "%.6f" % fit_time 
    print(info)
    info.to_csv("saved_models/Random_Search_Info.csv")


#stolen shamelessly off the internet
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))

            #except this part, this is the RMSLE score
            print("Mean RMSLE: {0:.3f} (std: {1:.3f})"
                  .format(np.sqrt(np.abs(results['mean_test_score'][candidate])),
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


if __name__ == "__main__":
 
    '''
    clean_method : integer
                   0 - deleted
                   1 - mean
                   2 - to_0
    
    preprocessing : integer
                    0 - ORIGINAL
                    1 - PCA
    '''
   # print(sys.argv)
    jobNo = int(sys.argv[1])     #For cluster
    clean_method = jobs[jobNo][1]  #For cluster
    preprocessing = jobs[jobNo][2]  #For cluster

    #read data from one of 6 datasets
    X, y = read_CSV(clean_method, preprocessing)

    '''
    algorithm : string
                - kNN
                - MLP
                - Decision Tree
                - SVM  
                - Random Forest
    '''
    algorithm = jobs[jobNo][0]   #"MLP"
    
    #this is where the params to test are stored
    param_dict = get_params(algorithm)

    #this where the actual searching happens
    random_search_(algorithm, param_dict, X, y, clean_method, preprocessing, iters=100, jobs=45)
   # print("Testing search with job params. Alg: %s, Clean Method: %d, Preprocessing: %d" %(jobs[jobNo][0],jobs[jobNo][1],jobs[jobNo][2]))
