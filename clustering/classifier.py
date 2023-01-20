import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def save_model(model,filename):
    joblib.dump(model, filename)


def logistic_regression():
    df=pd.read_csv('../Dataset/Clustered Jobs.csv')
    X = pd.read_csv('../Dataset/Clustered Components.csv')
    y = df['cluster_no']
