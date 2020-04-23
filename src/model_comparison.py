import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random as rand
import sys
import joblib
import matplotlib.pyplot as plt
import re
from sklearn.metrics import f1_score, classification_report
import time

train_labels = ["back","buffer_overflow","ftp_write","guess_passwd","imap","ipsweep","land","loadmodule","multihop","neptune","nmap","perl",
                "phf","pod","portsweep","rootkit","satan","smurf","spy","teardrop","warezclient","warezmaster","normal"]
test_labels = ["back","buffer_overflow","ftp_write","guess_passwd","imap","ipsweep","land","loadmodule","multihop","neptune","nmap","perl",
               "phf","pod","portsweep","rootkit","satan","smurf","spy","teardrop","warezclient","warezmaster","saint","snmpguess","mscan",
               "apache2","snmpgetattack","processtable","mailbomb","httptunnel","xlock","sendmail","named","xsnoop","ps","xterm","udpstorm",
               "sqlattack","worm","normal"]

subAttacks = {"dos":["back","land","neptune","pod","smurf","teardrop"],
              "r2l":["ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster"],
              "u2r":["buffer_overflow","loadmodule","perl","rootkit"],
              "probe":["ipsweep","nmap","portsweep","satan"] }

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
    overall_f1_score = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test,y_pred,labels=test_labels,output_dict=True)
    #binary_overall_f1_score = f1_score(y_test, y_pred, average='binary',pos_label='normal')
    if("accuracy" in report.keys()):
        report.pop("accuracy")
    #DEBUGGING######
    #print(report)
    #######DEBUGGING#####

    y_test_binary = []
    y_pred_binary = []    
    for elem in y_test:
        if elem == "normal":
            y_test_binary.append("normal")
        else:
            y_test_binary.append("attack")
    for elem in y_pred:
        if elem == "normal":
            y_pred_binary.append("normal")
        else:
            y_pred_binary.append("attack")

    #Sanity Check
    if(len(y_test_binary) != len(y_pred_binary)):
        print("BINARY NPY ARRAYS SHOULD BE SAME LENGTH!!")
        sys.exit() 

#    print("y_test_binary: %d   y_pred_binary: %d" % (len(y_test_binary),len(y_pred_binary)))

    overall_binary_f1_score = f1_score(y_test_binary, y_pred_binary, average='weighted')
    binary_report = classification_report(y_test_binary,y_pred_binary,labels=["normal","attack"],output_dict=True)

    if("accuracy" in binary_report.keys()):
        binary_report.pop("accuracy")

    y_test_subgroup = []
    y_pred_subgroup = []
    for elem in y_test:
        if elem == "normal":
            y_test_subgroup.append("normal")
        elif elem in subAttacks["dos"]:
            y_test_subgroup.append("dos")
        elif elem in subAttacks["r2l"]:
            y_test_subgroup.append("r2l")
        elif elem in subAttacks["u2r"]:
            y_test_subgroup.append("u2r")
        elif elem in subAttacks["probe"]:
            y_test_subgroup.append("probe")
        else:
            y_test_subgroup.append("other")

    for elem in y_pred:
        if elem == "normal":
            y_pred_subgroup.append("normal")
        elif elem in subAttacks["dos"]:
            y_pred_subgroup.append("dos")
        elif elem in subAttacks["r2l"]:
            y_pred_subgroup.append("r2l")
        elif elem in subAttacks["u2r"]:
            y_pred_subgroup.append("u2r")
        elif elem in subAttacks["probe"]:
            y_pred_subgroup.append("probe")
        else:
            y_pred_subgroup.append("other")

    #Sanity check
    if(len(y_pred_subgroup) != len(y_test_subgroup)):
        print("SUBGROUP NPY ARRAYS SHOULD BE SAME LENGTH!! y_test: %d  y_pred: %d" %(len(y_test_subgroup),len(y_pred_subgroup)))
        sys.exit()

    overall_subgroup_f1_score = f1_score(y_test_subgroup, y_pred_subgroup, average='weighted')
    subgroup_report = classification_report(y_test_subgroup,y_pred_subgroup,labels=["normal","dos","r2l","u2r","probe"],output_dict=True) 
    
    if("accuracy" in subgroup_report.keys()):
        subgroup_report.pop("accuracy")

    return overall_f1_score, report, overall_binary_f1_score,binary_report,overall_subgroup_f1_score,subgroup_report

#to make the skeleton
def make_report():

    algorithms = ["kNN", "MLP", "Decision_Tree", "SGD_Classifier", "Random_Forest","Naive-Bayes"]
    row_names = []
    for a in algorithms:
        name = "Best %s " % (a)
        row_names.append(name)

    col_names = ["PREDICTION TIME","OVERALL F1-SCORE"]
    for attack in test_labels:
        for val in ["precision","recall","f1-score","support"]:
            col_names.append("Overall_%s_%s" %(attack,val))
    for avg in ["micro avg", "macro avg","weighted avg"]:
        for val in ["precision","recall","f1-score","support"]: 
            col_names.append("Overall_%s_%s" %(avg,val)) 

    col_names.append("BINARY F1-SCORE")
    for binary in ["attack","normal"]:
        for val in ["precision","recall","f1-score","support"]: 
            col_names.append("Binary_%s_%s" %(binary,val)) 
    for avg in ["micro avg", "macro avg","weighted avg"]:
        for val in ["precision","recall","f1-score","support"]: 
            col_names.append("Binary_%s_%s" %(avg,val)) 

    col_names.append("ATTACK SUBGROUP F1-SCORE")
    for subgroup in ["dos","r2l","u2r","probe","normal","other"]:
        for val in ["precision","recall","f1-score","support"]: 
            col_names.append("Attack_Subgroup_%s_%s" %(subgroup,val)) 
    for avg in ["micro avg", "macro avg","weighted avg"]:
        for val in ["precision","recall","f1-score","support"]: 
            col_names.append("Attack_Subgroup_%s_%s" %(avg,val)) 

    df = pd.DataFrame(None, index=[row_names], columns=col_names)
    return(df)

#DO NOT RUN THIS AGAIN OR YOU'LL BE SORRY
def test_best_algs():
        
    algorithms = ["kNN", "MLP", "Decision_Tree", "SGD_Classifier", "Random_Forest","Naive-Bayes"]
    report = make_report()

    #bar_values = [None] * len(algorithms)
    models_path = "../saved_models/"
    for root, dirs, files in os.walk(models_path):
        for name in files:
            if name.endswith(".joblib"):
                #regex search for info from file name
                #info = re.findall("(?<=_)([^_]+)(?=_)", name)
                alg_name = name[5:-7]
                print(alg_name)
                clf = joblib.load(models_path + name) 
                X, y = load_test_data()
                print("Loaded %s%s" %(models_path,name))

                start_time = time.time()
                y_pred = clf.predict(X)
                end_time = time.time() - start_time
                overall_f1_score, overall_report, overall_binary_f1_score,binary_report,overall_subgroup_f1_score,subgroup_report = eval(y, y_pred)
                report.loc["Best %s " % (alg_name), "OVERALL F1-SCORE"] = "%.6f" % overall_f1_score
                report.loc["Best %s " % (alg_name), "BINARY F1-SCORE"] = "%.6f" % overall_binary_f1_score
                report.loc["Best %s " % (alg_name), "ATTACK SUBGROUP F1-SCORE"] = "%.6f" % overall_subgroup_f1_score
                report.loc["Best %s " % (alg_name), "PREDICTION TIME"] = "%.10f" % end_time
                for k1,v1 in overall_report.items():
                    for k2,v2 in v1.items():
                            report.loc["Best %s " % (alg_name),"Overall_%s_%s" % (k1,k2)] = "%.6f" % v2

                for k1,v1 in binary_report.items():
                    for k2,v2 in v1.items():
                            report.loc["Best %s " % (alg_name),"Binary_%s_%s" % (k1,k2)] = "%.6f" % v2

                for k1,v1 in subgroup_report.items():
                    for k2,v2 in v1.items():
                            report.loc["Best %s " % (alg_name),"Attack_Subgroup_%s_%s" % (k1,k2)] = "%.6f" % v2

    report.to_csv("Best_Model_Info.csv")
#################################    

if __name__ == "__main__":

    #Mean RMSLE or Refit Time
    #metric = "Mean RMSLE"
    
    #generate_rs_graphs(metric)
    test_best_algs()

    #Mean RMSLE or Prediction Time
    #metric = "Mean RMSLE"
    #generate_model_graphs(metric, lr=False)

