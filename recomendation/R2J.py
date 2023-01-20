from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity


def load_model(filename):
    return joblib.load(filename)


def get_recommendations(resume_text):
    # Vectorize user's skills and job descriptions
    vec=load_model('../Dataset/Model/skill_vector.sav')
    pca=load_model('../Dataset/Model/pca_vector.sav')
    model=load_model('../Dataset/Model/R2J_Logistic_Cls.sav')
    df=pd.read_csv('../Dataset/Clustered Jobs.csv')
    comps = pd.read_csv('../Dataset/Clustered Components.csv')
    
    skillz = pd.DataFrame(vec.transform([resume_text]).todense())
    skillz.columns = vec.get_feature_names()
    # Tranform feature matrix with pca
    user_comps = pd.DataFrame(pca.transform(skillz))
    print(user_comps.shape)
    # Predict cluster for user and print cluster number
    cluster = model.predict(user_comps)[0]
    print ('CLUSTER NUMBER', cluster, '\n\n')

    # Calculate cosine similarity
    cos_sim = pd.DataFrame(cosine_similarity(user_comps,comps[comps.index==cluster]))

    # Get job titles from sample2 to associate cosine similarity scores with jobs
    samp_for_cluster = df[df['cluster_no']==cluster]
    cos_sim = cos_sim.T.set_index(samp_for_cluster.position)
    cos_sim.columns = ['score']

    # Print the top ten suggested jobs for the user's cluster
    print ('Top ten suggested for your cluster', '\n', cos_sim.sort_values('score', ascending=False)[:10], '\n\n')

    return cos_sim