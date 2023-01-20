import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os


outdirmodel = './Dataset/Model'
outdirforcsv='./Dataset'

def load_model(filename):
    return joblib.load(filename)


def get_recommendations(resume_text):
    # Vectorize user's skills and job descriptions
    vec=load_model(os.path.join(outdirmodel,'skill_vector.sav'))
    pca=load_model(os.path.join(outdirmodel,'pca_vector.sav'))
    model=load_model(os.path.join(outdirmodel,'R2J_Logistic_Cls.sav'))
    df=pd.read_csv(os.path.join(outdirforcsv,'Clustered Jobs.csv'))
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    comps = pd.read_csv(os.path.join(outdirforcsv,'Clustered Components.csv'))
    comps.drop(comps.columns[comps.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    comps['cluster_no'] = df['cluster_no']
    comps.set_index('cluster_no', inplace=True)

    skillz = pd.DataFrame(vec.transform([resume_text]).todense())
    skillz.columns = vec.get_feature_names()
    # Tranform feature matrix with pca
    user_comps = pd.DataFrame(pca.transform(skillz))

    
    # Predict cluster for user and print cluster number
    cluster = model.predict(user_comps)[0]
    print ('CLUSTER NUMBER', cluster, '\n\n')

    # Calculate cosine similarity
    cos_sim = pd.DataFrame(cosine_similarity(user_comps,comps[comps.index==cluster]))
    # Get job titles from sample2 to associate cosine similarity scores with jobs
    samp_for_cluster = df[df['cluster_no']==cluster]
    # print(samp_for_cluster.head(10))
    cos_sim = cos_sim.T.set_index(samp_for_cluster.position)
    cos_sim.columns = ['score']

    # Print the top ten suggested jobs for the user's cluster
    # print ('Top ten suggested for your cluster', '\n', cos_sim.sort_values('score', ascending=False)[:10], '\n\n')

    return cos_sim