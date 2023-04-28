import numpy as np
from scipy.spatial.distance import cosine, euclidean, hamming
from spacy.lang.en import English
import pandas as pd
import json
import os

DEGREES_IMPORTANCE = {'high school': 0, 'associate': 1, 'BS-LEVEL': 2, 'MS-LEVEL': 3, 'PHD-LEVEL': 4}
ENTITIES = ['BS-LEVEL', 'MS-LEVEL', 'PHD-LEVEL', 'DEV', 'AI', 'CODING', 'DATA SCIENCES', 'AUTOMATION', 'BIG DATA',
            'WEB-DEVELOPMENT', 'MOBILE-DEVELOPMENT']
outdir = './Dataset/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

def assign_degree_match(match_scores):
        """calculate a degree matching score"""
        match_score = 0
        if len(match_scores) != 0:
            if max(match_scores) >= 2:
                match_score = 0.5
            elif (max(match_scores) >= 0) and (max(match_scores) < 2):
                match_score = 1
        return match_score

def degree_matching(resume, job):
        """calculate the final degree matching scores between resumes and job description"""
        score=1
        if len(job['degrees'])!=0:
            job_min_degree = DEGREES_IMPORTANCE[job['degrees']]
            match_scores = []
            for j in resume['degrees']:
                    score = DEGREES_IMPORTANCE[j] - job_min_degree
                    match_scores.append(score)
            score=assign_degree_match(match_scores)
        return score

 # majors matching
def get_major_category(major):
        """get a major's category"""
        with open('Resources/data/labels.json') as fp:
            labels = json.load(fp)
        categories = labels['MAJOR'].keys()
        for c in categories:
            if major in labels['MAJOR'][c]:
                return c

def get_job_acceptable_majors(job):
        """get acceptable job majors"""
        job_majors = job['majors']
        job_majors_categories = []
        for i in job_majors:
            job_majors_categories.append(get_major_category(i))
        return job_majors, job_majors_categories

def get_major_score(resume, job):
        """calculate major matching score for one resume"""
        resume_majors = resume['majors']
        job_majors, job_majors_categories = get_job_acceptable_majors(job)
        major_score = 0
        for r in resume_majors:
            if r in job_majors:
                major_score = 1
                break
            elif get_major_category(r) in job_majors_categories:
                major_score = 0.5
        return major_score

def major_matching(resume, job):
        """calculate major matching score for all resumes"""
        score=0
        if len(job['majors'])==0:
            score=1
        elif len(resume['majors'])!=0:
            score=get_major_score(resume ,job)
        return score

def calculate_single_cosine_similarity(vector1, vector2):
    return 1 - cosine(vector1, vector2)

def calculate_single_euclidean_similarity(vector1, vector2):
    return 1 / (1 + euclidean(vector1, vector2))

def calculate_single_hamming_similarity(vector1, vector2):
    return 1 - hamming(vector1, vector2)

def calculate_hybrid_similarity(vector1, vector2, cosine_weight=0.7, euclidean_weight=0.2, hamming_weight=0.1):
    cosine_similarity = calculate_single_cosine_similarity(vector1, vector2)
    euclidean_similarity = calculate_single_euclidean_similarity(vector1, vector2)
    hamming_similarity = calculate_single_hamming_similarity(vector1, vector2)

    # Set the weights for each similarity matrix
    similarity_weights = np.array([cosine_weight, euclidean_weight, hamming_weight])
    similarity_scores = np.array([cosine_similarity, euclidean_similarity, hamming_similarity])

    # Set negative, inf and nan scores to zero
    similarity_scores[np.logical_or(similarity_scores < 0, np.isinf(similarity_scores) == True)] = 0
    similarity_scores[np.isnan(similarity_scores)] = 0

    # Calculate the hybrid similarity score
    hybrid_similarity = np.dot(similarity_weights, similarity_scores)

    return hybrid_similarity