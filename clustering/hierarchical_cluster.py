from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import pandas as pd
import joblib
from sklearn.metrics import silhouette_score
from kneed import  KneeLocator
import os

outdir = './Dataset/Model'
outdirforcsv='./Dataset/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

def tokenizer(df):
    '''
        Parse the given skills dataframe to pull out appropriate skill phrases.
        Dataframe has some cells that are 2-gram nicely made skills, and other cells
        that are long runons with many skills.
        After scrubbing and then splitting on commas, we simplify the task by tossing
        out any greater than 4-gram phrases. 
    '''
    
    # Custom stop words that come up very often but don't say much about the job title.
    stops = ['manager', 'responsibilities', 'used', 'skills', 'duties', 'work', 'worked', 'daily',
             'services', 'job', 'using', '.com', 'end', 'prepare', 'prepared', 'lead', 'requirements','#39'] + list(stopwords.words('english'))
    values, ids, resume_ids = [],[],[]
    count = 0
    for idx, row in df.iterrows():
        # Split on commas
        array = row['skill'].lower().split(',')
        for x in array:
            # make sure the value is not empty or all numeric values or in the stop words list
            if x != '' and not x.lstrip().rstrip() in stops and not x.lstrip().rstrip().isdigit():
                # make sure single character results are the letter 'C' (programming language)
                if len(x) > 1 or x == 'C':
                    # drop stuff > 4 gram
                    if len(x.split(' ')) <= 4:
                        # update lists
                        
                        values.append(x.lstrip().rstrip())
                        ids.append(count)
                        count+=1
    
    # New dataframe with updated values.
    result_df = pd.DataFrame()
    result_df['skill'] = values
    return result_df

def save_model(model,filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

def calculate_optimal_cluster(model,X,limit):
    clusters = range(2, limit)
    wcss = []
    for i in clusters:
        hieratchical = AgglomerativeClustering(n_clusters=i)
        hieratchical.fit(X)
        score = silhouette_score(X, hieratchical.labels_)
        wcss.append(score)
    
    # create a knee locator object
    kl = KneeLocator(clusters, wcss, curve='convex', direction='decreasing')

    # get the optimal number of clusters
    optimal_clusters = kl.elbow
    return optimal_clusters

def create_cluster(df):
    test_df = tokenizer(df)
    voc = test_df['skill'].unique()
    vec = TfidfVectorizer(vocabulary=voc, decode_error='ignore')
    skills_matrix = vec.fit_transform(df['skill'])
    save_model(vec,os.path.join(outdir,'skill_vector.sav') )
    skills_matrix = pd.DataFrame(skills_matrix.todense())
    skills_matrix.columns = vec.get_feature_names()
    # Run PCA to reduce number of features
    pca = PCA(n_components=len(skills_matrix), random_state=42)
    comps = pca.fit_transform(skills_matrix)
    print("Comps :",comps.shape)
    save_model(pca,os.path.join(outdir,'pca_vector.sav') )
    # Put the components into a dataframe
    comps = pd.DataFrame(comps)
    print(len(skills_matrix))
    comps.to_csv(os.path.join(outdirforcsv,'Clustered Components.csv'))
    # Cluster job titles based on components derived from feature matrix
    k=calculate_optimal_cluster('hierarchical',comps,15)
    cltr = AgglomerativeClustering(n_clusters=k)
    cltr.fit(comps)
    # Add new column containing cluster number to sample, comps, and feature matrix dataframes
    df['cluster_no'] = cltr.labels_
    df.to_csv(os.path.join(outdirforcsv,'Clustered Jobs.csv') )


def main():
    df=pd.read_csv(os.path.join(outdirforcsv, 'Skill Extract.csv'))
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    df=df.reset_index(drop=True)
    create_cluster(df)



if __name__ == '__main__':
      main()
      print('-----------Hirerarchical clustering of job data is complete. Check the csv file.-----------')
    