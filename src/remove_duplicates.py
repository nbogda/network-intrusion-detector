import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# helper function to cast items to float where ever possible
def float_cast(item):
    try:
        return float(item)
    except ValueError:
        return item

def main():

    # converts every row in the train data to a string, then adds it to a set
    # if the same string row already exists in the set, then it will not be added
    f=open("../data/train_data","r")
    line_set={"0"}
    while True:
        line=f.readline()
        if line =="":
            break
        line=line[:-2]
        line_set.add(line)
    line_set.remove("0")
    
    # goes through the unduplicated string set now and splits the data by comma
    # converts to float where possible
    list_data = []
    for line in line_set:
        line = line.split(',')
        line = [float_cast(l) for l in line]
        list_data.append(line)

    # changes all of the categorical variables into ordinally encoded variables
    # saves this result into csv file to be used for predictions
    list_data = pd.DataFrame(list_data)
    enc = OrdinalEncoder()
    list_data.iloc[:,1:4] = enc.fit_transform(list_data.iloc[:,1:4])
    list_data.to_csv("../data/train_nodup.csv",index=False)  

if __name__ == "__main__":
    exit(main())
