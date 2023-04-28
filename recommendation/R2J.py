import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
import matching_rule as utils
import numpy as np

outdirmodel = './Dataset/Model'
outdirforcsv = './Dataset'


def load_model(filename):
    return joblib.load(filename)

def give_recommendations(resume,similarity_weights):
    # Vectorize user's skills and job descriptions
    recommendation = {}
    vec = load_model(os.path.join(outdirmodel, 'skill_vector.sav'))
    pca = load_model(os.path.join(outdirmodel, 'pca_vector.sav'))
    model = load_model(os.path.join(outdirmodel, 'R2J_Logistic_Cls.sav'))
    df = pd.read_csv(os.path.join(outdirforcsv, 'Clustered Jobs.csv'))
    df=df.reset_index(drop=True)
    df.drop(df.columns[df.columns.str.contains(
        'unnamed', case=False)], axis=1, inplace=True)
    total_cluster = df['cluster_no'].nunique()
    recommendation['total_cluster'] = int(total_cluster)
    comps = pd.read_csv(os.path.join(outdirforcsv, 'Clustered Components.csv'))
    comps.drop(comps.columns[comps.columns.str.contains(
        'unnamed', case=False)], axis=1, inplace=True)
    comps=comps.reset_index(drop=True)
    comps['cluster_no'] = df['cluster_no']
    skillz = pd.DataFrame(vec.transform([resume['skill']]).todense())
    skillz.columns = vec.get_feature_names()
    # Tranform feature matrix with pca
    user_comps = pd.DataFrame(pca.transform(skillz))
    # Predict cluster for user and print cluster number
    cluster = model.predict(user_comps)[0]
    recommendation['predicted_cluster_no'] =int( cluster)
    # Calculate the similarity scores between the resume and each job description
    all_cluster_jobs = []
    for i in range(total_cluster):
        sam_coms=comps[comps['cluster_no']==i]
        samp_for_cluster = df[df['cluster_no']==i]
        similarity_scores = []
        for index, job in sam_coms.iterrows():
            job=job.drop('cluster_no')
            semantic_similarity = utils.calculate_hybrid_similarity(user_comps.T,job.T)
            major_score=utils.major_matching(resume,samp_for_cluster.loc[index])
            degree_score=utils.degree_matching(resume,samp_for_cluster.loc[index])
            total_similarity = np.array([semantic_similarity, major_score, degree_score])
            final_score=np.dot(similarity_weights, total_similarity)
            similarity_scores.append(final_score)
        similarity = pd.DataFrame(similarity_scores)
        similarity = similarity.set_index(samp_for_cluster.position)
        similarity.columns = ['score']
        similarity = similarity.join(df.set_index(df.position))
        # Sort the job dataframe by similarity score in descending order
        similarity = similarity.sort_values(
            'score', ascending=False).reset_index(drop=True).head(10)
        all_cluster_jobs.append(cluster.to_dict(orient='records'))
    recommendation['all_cluster_jobs']=all_cluster_jobs
    return recommendation


def get_recommendations(resume_text):
    # Vectorize user's skills and job descriptions
    recommendation = {}
    vec = load_model(os.path.join(outdirmodel, 'skill_vector.sav'))
    pca = load_model(os.path.join(outdirmodel, 'pca_vector.sav'))
    model = load_model(os.path.join(outdirmodel, 'R2J_Logistic_Cls.sav'))
    df = pd.read_csv(os.path.join(outdirforcsv, 'Clustered Jobs.csv'))
    total_cluster = df['cluster_no'].nunique()
    recommendation['total_cluster'] = int(total_cluster)
    df.drop(df.columns[df.columns.str.contains(
        'unnamed', case=False)], axis=1, inplace=True)
    comps = pd.read_csv(os.path.join(outdirforcsv, 'Clustered Components.csv'))
    comps.drop(comps.columns[comps.columns.str.contains(
        'unnamed', case=False)], axis=1, inplace=True)
    comps['cluster_no'] = df['cluster_no']
    comps.set_index('cluster_no', inplace=True)
    skillz = pd.DataFrame(vec.transform([resume_text]).todense())
    skillz.columns = vec.get_feature_names()
    # Tranform feature matrix with pca
    user_comps = pd.DataFrame(pca.transform(skillz))
    # Predict cluster for user and print cluster number
    cluster = model.predict(user_comps)[0]
    recommendation['predicted_cluster_no'] =int( cluster)
    # Calculate cosine similarity
    cos_sim = pd.DataFrame(cosine_similarity(
        user_comps, comps[comps.index == cluster]))
    # Get job titles from sample2 to associate cosine similarity scores with jobs
    samp_for_cluster = df[df['cluster_no'] == cluster]
    # print(samp_for_cluster.head(10))
    cos_sim = cos_sim.T.set_index(samp_for_cluster.position)
    cos_sim.columns = ['score']
    df.set_index(df.position, inplace=True)
    recommend_jobs = cos_sim.join(df)
    recommend_jobs = recommend_jobs.sort_values(
        'score', ascending=False).reset_index(drop=True).head(5)
  
    records = recommend_jobs.to_dict(orient='records')
    recommendation['recommend_jobs'] = records

    # Print the top five suggested jobs for each cluster
    all_cluster_jobs = []

    skillz = skillz.T
    for i in range(total_cluster):
        cos_sim = pd.DataFrame(cosine_similarity(
            user_comps, comps[comps.index == i]))
        samp_for_cluster = df[df['cluster_no'] == i]
        cos_sim = cos_sim.T.set_index(samp_for_cluster.position)
        cos_sim.columns = ['score']
        cluster = cos_sim.join(df)
        cluster = cluster.sort_values(
            'score', ascending=False).reset_index(drop=True).head(5)
        all_cluster_jobs.append(cluster.to_dict(orient='records'))
    recommendation['all_cluster_jobs']=all_cluster_jobs

    # Print the top ten suggested jobs for the user's cluster
    # print ('Top ten suggested for your cluster', '\n', cos_sim.sort_values('score', ascending=False)[:10], '\n\n')

    return recommendation
