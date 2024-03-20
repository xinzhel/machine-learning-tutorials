import pandas as pd
from matplotlib import pyplot as plt

def load_digits(file1, file2=None): 
    
    if file2 is None:
        df = pd.read_csv(file1, header=None) # file1: "data/digitData2.csv"
        X = df.iloc[:, :-1] # features
        y = df.iloc[:, -1] # labels
        return X, y
    else:
        digit_zero=pd.read_csv(file1,header=None) 
        print('digit zero:', digit_zero.size)
        digit_one=pd.read_csv(file2,header=None)
        print('digit one:', digit_one.size)


        # merge two files
        df=pd.concat([digit_zero,digit_one],join="inner")
        df.columns=["feature_"+str(i+1) for i in range(df.shape[1]-1)] + ["target"]

        # we only select digits 0, 1
        
        df['target'] = df['target'].astype(int)
        df_selected = df[(df.target==0) | (df.target==1)]

        # split features and labels
        X=df_selected.iloc[:,:-1]
        y=df_selected.iloc[:,-1]
        return X, y
