#this is just to make the skeleton of the dataframe to store info from the grid search
import pandas as pd

file_paths = ["deleted", "mean", "to_0"]
file_names = ["ORIGINAL", "PCA"]
algorithms = ["kNN", "MLP", "Decision Tree", "SVM", "Random Forest"]

row_names = []

for a in algorithms:
    for n in file_names:
        for p in file_paths:
            name = "Best %s %s %s" % (a, n, p)
            row_names.append(name)

col_names = ["Best Params", "Mean RMSLE", "Refit Time"]

df = pd.DataFrame(None, index=[row_names], columns=col_names)
df.to_csv("saved_models/Random_Search_Info.csv")
