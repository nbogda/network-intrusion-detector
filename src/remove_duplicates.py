import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def main():

    f=open("../data/train_data","r")
    line_set={"0"}
    while True:
        line=f.readline()
        if line =="":
            break
        line=line[:-2]
        line_set.add(line)
    line_set.remove("0")
    
    line_set = pd.DataFrame(list(line_set))
    enc = OrdinalEncoder()
    line_set.iloc[:,1:4] = enc.fit_transform(line_set.iloc[:,1:4])
    line_set.to_csv("../data/train_nodup.csv",index=False)  

if __name__ == "__main__":
    exit(main())
