import pandas as pd
import joblib
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


import json

def get_job_by_id(job_id):
    df = pd.read_csv(os.path.join(outdirforcsv, 'Clustered Jobs.csv'))
    df = df.fillna("")
    df = df.reset_index(drop=True)
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df = df[['position', 'company', 'description', 'skill', 'link', 'Minimum degree level', 'Acceptable majors','cluster_no']]
    job = df[df['position'] == job_id]
    job_json = job.to_json(orient='records')
    job_object = json.loads(job_json)[0]
    return job_object

