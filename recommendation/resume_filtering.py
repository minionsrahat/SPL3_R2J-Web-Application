import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from recommendation import matching_rule as utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
from resume_screening import resume_parser
from clustering import newclustering_method

outdirmodel = './Dataset/Model'
outdirforcsv = './Dataset'

def load_model(filename):
    return joblib.load(filename)

def resume_filtering(resumes,job_des,similarity_weights):
    # Vectorize user's skills and job descriptions
    vocab = newclustering_method.build_vocab(job_des)
    # Define the vocabulary and vectorizer
    voc = list(vocab.word2idx.keys())
    vec = TfidfVectorizer(vocabulary=voc, decode_error='ignore')
    # Vectorize the resume descriptions
    resume_desc_matrix = vec.fit_transform(resumes['skills'])
    # Convert the matrix to a dataframe with feature names
    resume_desc_matrix = pd.DataFrame(resume_desc_matrix.todense())
    resume_desc_matrix.columns = vec.get_feature_names()
    # Run PCA to reduce number of features
    pca = PCA(n_components=len(resume_desc_matrix.columns), random_state=42)
    comps = pca.fit_transform(resume_desc_matrix)
    # Put the components into a dataframe with feature names
    comps = pd.DataFrame(comps)
    skillz = pd.DataFrame(vec.transform([job_des['skills']]).todense())
    skillz.columns = vec.get_feature_names()
        # Tranform feature matrix with pca
    job_comps = pd.DataFrame(pca.transform(skillz))
    similarity_scores = []
    for index, item in resumes.iterrows():
            semantic_similarity = utils.calculate_hybrid_similarity(item.T,job_comps.T)
            major_score=utils.major_matching(resumes.loc[index],job_des)
            degree_score=utils.degree_matching(resumes.loc[index],job_des)
            total_similarity = np.array([semantic_similarity, major_score, degree_score])
            final_score=np.dot(similarity_weights, total_similarity)
            similarity_scores.append(final_score)
    similarity = pd.DataFrame(similarity_scores)
    similarity = similarity.set_index(resumes.index)
    similarity.columns = ['similarity_score']
    similarity = similarity.join(resumes)
        # Sort the job dataframe by similarity score in descending order
    similarity = similarity.sort_values(
            'similarity_score', ascending=False).reset_index(drop=True)
    return similarity


def main():
     resume_file="Arbi Dwi.pdf"
    #  details=resume_parser.parser(resume_file)
    #  recommendation=give_recommendations(details,[0.8,0.1,0.1])
    #  print(recommendation)
     
if __name__ == '__main__':
      main()
      print('-----------Job Recommendation.-----------')
    