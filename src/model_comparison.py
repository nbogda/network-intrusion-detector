import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random as rand
import sys
import joblib
import matplotlib.pyplot as plt
import re
from sklearn.metrics import f1_score
import time


#graphs to visualize data from the random search
def generate_rs_graphs(metric, rf=True):
    '''
    metric : string
             Mean RMSLE or Refit Time
    
    rf     : boolean
             if rf=False, will exclude Random Forest from graph
             only to be used when metric="Refit Time"
    '''
    random_search_info = pd.read_csv("param_search/saved_models/Random_Search_Info.csv", index_col=0)
   
    #to exclude or not to exclude random forest? 
    N = 6
    algorithms = ["kNN", "MLP", "Decision_Tree", "SGD_Classifier", "Random_Forest","Naive-Bayes"]
    if metric == "Refit Time" and not rf:
        algorithms = ["kNN", "MLP", "Decision_Tree", "SGD_Classifier","Naive-Bayes"]
        N = 5
   
    #make arrays to hold info for the 6 datasets
    bar_values = [None] * N
    for index, row in random_search_info.iterrows():
        #regex search for algorithm name in pandas file
        name = re.search("Best (.*) [^\s]+ [^\s]+", row.name).group(1)
        if not rf and name == "Random_Forest":
            continue
        #add data to appropriate index
        bv_index = algorithms.index(name)
        if bar_values[bv_index] is None:
            bar_values[bv_index] = []
        bar_values[bv_index].append(row.loc[metric])
    
    #plot the bars
    fig, ax = plt.subplots(figsize=(20,10))
    ind = np.arange(N)
    width = 0.1
    bar_values = np.array(bar_values)
    og_del = ax.bar(ind, bar_values[:,0], width)
    og_mean = ax.bar(ind + width, bar_values[:,1], width)
    og_0 = ax.bar(ind + width*2, bar_values[:,2], width)
    pca_del = ax.bar(ind + width*3, bar_values[:,3], width)
    pca_mean = ax.bar(ind + width*4, bar_values[:,4], width)
    pca_0 = ax.bar(ind + width*5, bar_values[:,5], width)
    if metric == "Refit Time":
        metric += " (s)"
    ax.set_ylabel(metric, fontsize=14)
    ax.set_xticks((ind + width*2.5))
    ax.tick_params(labelsize=14)
    ax.set_xticklabels(algorithms)
    ax.legend((og_del[0], og_mean[0], og_0[0], pca_del[0], pca_mean[0], pca_0[0]), 
              ("Original NaN Deleted", "Original Mean Impute", "Original 0 Impute", "PCA NaN Deleted", "PCA Mean Impute", "PCA 0 Impute"),
              ncol=2, fontsize='large')
    title = "Performance" if metric == "Mean RMSLE" else "Fit Time"
    if not rf:
        title += " without Random Forest"
    ax.set_title("Random Search Best Algorithm %s" % title, fontsize=14)
    plt.savefig("graphs/random_search_best_alg_%s.png" % title)

#graphs to visualize data from the random search
def generate_model_graphs(metric, lr=True):
    '''
    metric : string
             Mean RMSLE or Prediction Time
    '''
    graph_info = pd.read_csv("graphs/Best_Model_Info.csv", index_col=0)
   
    algorithms = ["kNN", "MLP", "Decision Tree", "SVM", "Random Forest", "linearRegression"]
    if not lr:
        algorithms = ["kNN", "MLP", "Decision Tree", "SVM", "Random Forest"]
    N = len(algorithms)

    #peter prefenhoffer
    top_score = 0.55494
    
    #make arrays to hold info for the 6 datasets
    bar_values = [None] * N 
    for index, row in graph_info.iterrows():
        #regex search for algorithm name in pandas file
        name = re.search("Best (.*) [^\s]+ [^\s]+", row.name).group(1)
        if name == "linearRegression" and not lr:
            continue
        #add data to appropriate index
        bv_index = algorithms.index(name)
        if bar_values[bv_index] is None:
            bar_values[bv_index] = []
        bar_values[bv_index].append(row.loc[metric])
    
    #plot the bars
    fig, ax = plt.subplots(figsize=(20,10))
    ind = np.arange(N)
    width = 0.1
    bar_values = np.array(bar_values)
    og_del = ax.bar(ind, bar_values[:,0], width)
    og_mean = ax.bar(ind + width, bar_values[:,1], width)
    og_0 = ax.bar(ind + width*2, bar_values[:,2], width)
    pca_del = ax.bar(ind + width*3, bar_values[:,3], width)
    pca_mean = ax.bar(ind + width*4, bar_values[:,4], width)
    pca_0 = ax.bar(ind + width*5, bar_values[:,5], width)
    if metric == "Prediction Time":
        metric += " (s)"
    ax.set_ylabel(metric, fontsize=14)
    ax.set_xticks((ind + width*2.5))
    ax.tick_params(labelsize=14, length=6, width=2)
    if lr:
        algorithms = ["kNN", "MLP", "Decision Tree", "SVM", "Random Forest", "Linear Regression"]
    ax.set_xticklabels(algorithms)
    legend1 = ax.legend((og_del[0], og_mean[0], og_0[0], pca_del[0], pca_mean[0], pca_0[0]), 
              ("Original NaN Deleted", "Original Mean Impute", "Original 0 Impute", "PCA NaN Deleted", "PCA Mean Impute", "PCA 0 Impute"),
              ncol=2, fontsize='large')
    line = plt.axhline(y=top_score, linestyle="--")
    loc_ = (0.001, 0.82)
    if not lr:
        loc_ = (0.86, 0.82)
    ax.legend([line], ["Top Kaggle Score"], fontsize='large', loc=loc_)
    plt.gca().add_artist(legend1)
    title = "Prediction Performance" if metric == "Mean RMSLE" else "Prediction Time"
    if not lr:
        title += " without Linear Regression"
    ax.set_title("Test Set Best Algorithm %s" % title, fontsize=14)
    plt.savefig("graphs/test_set_best_alg_%s" % title)

################################################### Used to TEST models #############

def load_test_data():

    X = np.load("../data/X_test_PCA.npy")
    y = np.load("../data/y_test_PCA.npy")

    return X, y

def eval(y_test, y_pred):
    score = f1_score(y_test, y_pred, average='weighted')
    return score

#to make the skeleton
def make_report():

    algorithms = ["kNN", "MLP", "Decision_Tree", "SGD_Classifier", "Random_Forest","Naive-Bayes"]
    row_names = []
    for a in algorithms:
        name = "Best %s " % (a)
        row_names.append(name)
    col_names = ["Mean F1-Score", "Prediction Time"]
    #col_names += ["RMSLE Month %d" % i for i in range(1, 13)]  TODO: Maybe use this for F1-Score for binary??
    df = pd.DataFrame(None, index=[row_names], columns=col_names)
    return(df)

#DO NOT RUN THIS AGAIN OR YOU'LL BE SORRY
def test_best_algs():
        
    algorithms = ["kNN", "MLP", "Decision_Tree", "SGD_Classifier", "Random_Forest","Naive-Bayes"]
    report = make_report()

    bar_values = [None] * len(algorithms)
    models_path = "param_search/saved_models/"
    for root, dirs, files in os.walk(models_path):
        for name in files:
            if name.endswith(".joblib"):
                #regex search for info from file name
                info = re.findall("(?<=_)([^_]+)(?=_)", name)
                                #clean_method = re.search(".*_(.*).joblib", name).group(1)
                alg_name = info[0]
                                #preprocess = info[1]
                                #if clean_method == "0": clean_method = "to_0"
                clf = joblib.load(models_path + name) 
                X, y = load_test_data()
                print("Loaded %s%s" %(models_path,name))

                #MAYBE HAVE THIS PREDICT NORMAL BUT THEN ALSO BINARY!!

                start_time = time.time()
                y_pred = clf.predict(X)
                end_time = time.time() - start_time
                overall = eval(y, y_pred)
                report.loc["Best %s" % (alg_name), "Mean F1-Score"] = "%.6f" % overall 
                report.loc["Best %s" % (alg_name), "Prediction Time"] = "%.10f" % end_time
                                #for i in range(0, len(multiple)):
                                    #report.loc["Best %s %s %s" % (alg_name, preprocess, clean_method), "RMSLE Month %s" % str(i + 1)] = "%.6f" % multiple[i]
    report.to_csv("graphs/Best_Model_Info.csv")
#################################    

if __name__ == "__main__":

    #Mean RMSLE or Refit Time
    #metric = "Mean RMSLE"
    
    #generate_rs_graphs(metric)
    test_best_algs()

    #Mean RMSLE or Prediction Time
    #metric = "Mean RMSLE"
    #generate_model_graphs(metric, lr=False)

