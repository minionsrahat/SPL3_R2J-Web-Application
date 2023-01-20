import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

outdir = './Dataset/Model'
if not os.path.exists(outdir):
    os.mkdir(outdir)

def save_model(model,filename):
    joblib.dump(model, filename)

def logistic_regression():
    df=pd.read_csv('../Dataset/Clustered Jobs.csv')
    X = pd.read_csv('../Dataset/Clustered Components.csv')
    y = df['cluster_no']
    lr = LogisticRegression(C=10, penalty='l2', multi_class='multinomial', solver='sag', max_iter=1000)
    lr.fit(X, y)
    save_model(lr,os.path.join(outdir,'R2J_Logistic_Cls.sav'))
