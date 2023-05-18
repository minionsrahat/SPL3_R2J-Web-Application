import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from recommendation import matching_rule as utils
import numpy as np
from resume_screening import resume_parser

outdirmodel = './Dataset/Model'
outdirforcsv = './Dataset'

def load_model(filename):
    return joblib.load(filename)


def get_cluster_wise_jobs():
     df = pd.read_csv(os.path.join(outdirforcsv, 'Clustered Jobs.csv'))
     df = df.fillna("")
     df=df.reset_index(drop=True)
     df.drop(df.columns[df.columns.str.contains(
        'unnamed', case=False)], axis=1, inplace=True)
     total_cluster = df['cluster_no'].nunique()
     jobs={}
     all_cluster_jobs = []
     for i in range(total_cluster):
        samp_for_cluster = df[df['cluster_no']==i]
        samp_for_cluster=samp_for_cluster.head(100)
        all_cluster_jobs.append(samp_for_cluster.to_dict(orient='records'))
     jobs['all_cluster_jobs']=all_cluster_jobs

