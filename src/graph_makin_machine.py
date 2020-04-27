# put all yer graphin stuff right in heeyah
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import random as rand
import sys
import collections
import joblib
import seaborn as sn
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from itertools import combinations, combinations_with_replacement
from sklearn.metrics import make_scorer, f1_score, roc_curve, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# for charlie's best models based on random_search_info.csv
def best_models():
    model_dict = {  
                    "Naive-Bayes" : GaussianNB(),
                    "SGD_Classifier" : SGDClassifier(max_iter=1000, alpha=0.0001),
                    "Decision Tree" : DecisionTreeClassifier(min_samples_split=4, min_samples_leaf=1, max_features='log2', criterion='entropy'),
                    "kNN" : KNeighborsClassifier(p=1, n_neighbors=1),
                    "MLP" : MLPClassifier(learning_rate_init=0.001, hidden_layer_sizes=(10, 10, 10, 20, 20), batch_size=200, alpha=0.01, activation='tanh'),
                    "Random_Forest" : RandomForestClassifier(n_estimators=50, min_samples_split=2, min_samples_leaf=1, max_features='log2', criterion='entropy')
                  }

    return model_dict
 
# the x_train and x_test here are the actual train and test sets
# NOT the split train set
def get_data():
    X_train = np.load("data_preprocessing/X_train_PCA.npy")
    y_train = np.load("data_preprocessing/y_train_PCA.npy")
    X_test = np.load("data_preprocessing/X_test_PCA.npy")
    y_test = np.load("data_preprocessing/y_test_PCA.npy")
    return (X_train, y_train, X_test, y_test)

# just counting the labels in train and test set
# output table of labels
def read_labels(data):
    train_label_dict = {}
    test_label_dict = {}
    unseen = []

    train_labels = data[1]
    test_labels = data[3]
    test_data = data[2]

    # count different train labels
    for t in train_labels:
        if t not in train_label_dict:
            train_label_dict[t] = 0
        train_label_dict[t] += 1

    # count different test labels
    for t, d in zip(test_labels, test_data):
        if t not in test_label_dict:
            test_label_dict[t] = []
        test_label_dict[t].append(d)

    # count unseen
    for k in test_label_dict.keys():
        if k not in train_label_dict:
            for attack in test_label_dict[k]:
                unseen.append(attack)
    return unseen
    
    '''
    # apparently the test set doesn't have some stuff thats in the training set
    for k in train_label_dict.keys():
        if k not in test_label_dict:
            test_label_dict[k] = 0
            if 'not_in_test' not in unseen:
                unseen['not_in_test'] = 0
            unseen['not_in_test'] += train_label_dict[k]
    
    unseen_list = pd.DataFrame([unseen], index=["Instances of Attacks"])
    unseen_list = unseen_list.transpose()
    # output csv file, this should go in report
    unseen_list.to_csv('graphs/unseen_attacks_list.csv')

    attacks_list = pd.DataFrame([train_label_dict, test_label_dict], index=["Train", "Test"])
    attacks_list = attacks_list.transpose().sort_values(by='Train', ascending=False)
    # output csv file, this should go in report
    attacks_list.to_csv('graphs/attacks_list.csv')
    '''
# binarize labels and output counts
# test accuracy of model and plot ROC curve
def plot_ROC(data, model_dict):

    train_labels = []
    test_labels = []
    
    # just curious
    # train_dict = {}
    # test_dict = {}
    
    for x in data[1]:
        if x != "normal":
             x = "attack"
             # x = 1
        #else:
             # x = 0
        # if x not in train_dict:
            # train_dict[x] = 0
        # train_dict[x] += 1
        train_labels.append(x)

    for x in data[3]:
        if x != "normal":
            x = "attack"
            # x = 1
        # else:
            # x = 0
        # if x not in test_dict:
            # test_dict[x] = 0
        # test_dict[x] += 1
        test_labels.append(x)

    # bin_attacks_list = pd.DataFrame([train_dict, test_dict], index=["Train", "Test"])
    # bin_attacks_list = bin_attacks_list.transpose().sort_values(by='Train', ascending=False)
    # output csv file, this should go in report
    # bin_attacks_list.to_csv('graphs/binary_attacks_list.csv')

    # go through all models and plot ROC curves
    # fpr_tpr_auc = []
    for k, v in model_dict.items():
        y_pred_proba = None
        model = v.fit(data[0], train_labels)
        # try:
        print("Predicting %s" % k)
        y_pred = model.predict(data[2])
        confusion_matrix_(test_labels, y_pred, k)
            
            # y_pred_proba = model.predict_proba(data[2])
            
        # except AttributeError as e:
            # print("Failed, predicting %s" % k)
            # y_pred_proba = model.decision_function(data[2])
        
        '''
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, _ = roc_curve(test_labels, y_pred_proba)
        roc_auc = roc_auc_score(test_labels, y_pred_proba)
        fpr_tpr_auc.append([k, fpr, tpr, roc_auc])
        '''
    # ROC_curve(fpr_trp_auc)

def plot_unseen(model_dict, unseen, data):
    
    train_labels = []
    accuracies = {}
    
    for x in data[1]:
        if x != "normal":
             x = "attack"
        train_labels.append(x)

    for k, v in model_dict.items():
        y_pred_proba = None
        model = v.fit(data[0], train_labels)
        print("Predicting %s" % k)
        y_pred = model.predict(unseen)
        accuracy = (y_pred == "attack").sum()/len(y_pred) # % of attack predicted correctly
        accuracies[k] = accuracy
    
    index = np.arange(len(accuracies.keys()))
    fig = plt.figure(figsize=(9, 7))
    plt.bar(index, accuracies.values())
    plt.xlabel('Classifier')
    plt.ylabel('Percent Correct')
    plt.xticks(index, accuracies.keys())
    plt.title('Percent of Unseen Attacks Correctly Identified (Total = 4,022)')
    plt.savefig("graphs/unseen_predictions.png")

    


def confusion_matrix_(y_test, y_pred, k):
    labels = np.unique(y_test)

    print(y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="attack")

    # get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(9, 7))
    sn.heatmap(cm, annot=True, ax=ax, fmt='g', cmap='Blues', linewidths=0.25, linecolor='black')
    ax.set_xlabel('Predicted Labels', fontsize='large')
    ax.set_ylabel('True Labels', fontsize='large')
    ax.set_title("Confusion matrix for %s, F1 Score = %.4f" % (k, f1), fontsize='large')
    ax.xaxis.set_ticklabels(labels, rotation='horizontal', fontsize='large')
    ax.yaxis.set_ticklabels(labels, rotation='horizontal', fontsize='large')
    plt.savefig("graphs/%s_confusion_matrix.png" % k)


    
def ROC_curve(fpr_tpr, auc):
    # plot ROC curve
    for d in fpr_tpr_auc:
        plt.plot(d[1], d[2], label="%s, AUC = %.4f" % (d[0], d[3]))
    plt.plot([0, 1], [0, 1], 'k--')  # random 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('Best Models ROC Curve')
    plt.legend()
    plt.savefig('graphs/roc_curves.png')


def main():

    model_dict = best_models()
    data = get_data()
    unseen = read_labels(data)
    plot_unseen(model_dict, unseen, data)
    # plot_ROC(data, model_dict)
    



if __name__ == "__main__":
    main()
