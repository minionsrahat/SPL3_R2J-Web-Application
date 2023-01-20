import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

outdirmodel = './Dataset/Model'
outdirforcsv='./Dataset'
if not os.path.exists(outdirmodel):
    os.mkdir(outdirmodel)

def save_model(model,filename):
    joblib.dump(model, filename)

def logistic_regression():
    df=pd.read_csv(os.path.join(outdirforcsv,'Clustered Jobs.csv'))
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    X = pd.read_csv(os.path.join(outdirforcsv,'Clustered Components.csv'))
    X.drop(X.columns[X.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    print("Comps :",X.shape)
    y = df['cluster_no']
    lr = LogisticRegression(C=10, penalty='l2', multi_class='multinomial', solver='sag', max_iter=1000)
    lr.fit(X, y)
    save_model(lr,os.path.join(outdirmodel,'R2J_Logistic_Cls.sav'))

def main():
    logistic_regression()
   
if __name__ == '__main__':
      main()
      print('-----------Classifier of clustered job data is complete-----------------')