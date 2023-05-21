from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import nltk
import pandas as pd
import joblib
from sklearn.metrics import silhouette_score
from kneed import  KneeLocator
from collections import Counter
import os

outdir = './Dataset/Model'
outdirforcsv='./Dataset/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(df, threshold=1):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    
    for i in range(len(df)):
        caption = df.loc[i, 'skills']
        tokens = nltk.tokenize.word_tokenize(str(caption))
        counter.update(tokens)

        if (i+1) % 1000 == 0:
                print("[{}/{}] Tokenized the sentences.".format(i+1, len(df)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def save_model(model,filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

def calculate_optimal_cluster(model,X,limit):
    # Range of cluster numbers to test
    cluster_range = range(2, limit)

    # List to store silhouette scores
    silhouette_scores = []

    # Perform Agglomerative Clustering for each cluster number and calculate silhouette score
    for n_clusters in cluster_range:
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agglomerative.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    # create a knee locator object
    # create a knee locator object
    kl = KneeLocator(cluster_range, silhouette_scores, curve='convex')

    # get the optimal number of clusters
    optimal_clusters = kl.elbow
    return optimal_clusters

def create_cluster(df):
    vocab = build_vocab(df)
    # Define the vocabulary and vectorizer
    voc = list(vocab.word2idx.keys())
    vec = TfidfVectorizer(vocabulary=voc, decode_error='ignore')
    # Vectorize the job descriptions
    job_desc_matrix = vec.fit_transform(df['skills'])
    # Convert the matrix to a dataframe with feature names
    job_desc_matrix = pd.DataFrame(job_desc_matrix.todense())
    job_desc_matrix.columns = vec.get_feature_names()
    # Run PCA to reduce number of features
    pca = PCA(n_components=len(job_desc_matrix.columns), random_state=42)
    comps = pca.fit_transform(job_desc_matrix)
    # Put the components into a dataframe with feature names
    comps = pd.DataFrame(comps)
    save_model(vec,os.path.join(outdir,'skill_vector.sav') )
    save_model(pca,os.path.join(outdir,'pca_vector.sav') )
    comps.to_csv(os.path.join(outdirforcsv,'Clustered Components.csv'))
    # Cluster job titles based on components derived from feature matrix
    k=calculate_optimal_cluster('hierarchical',comps,15)
    cltr = AgglomerativeClustering(n_clusters=k)
    cltr.fit(comps)
    # Add new column containing cluster number to sample, comps, and feature matrix dataframes
    df['cluster_no'] = cltr.labels_
    df.to_csv(os.path.join(outdirforcsv,'Clustered Jobs.csv') )

def main():
    df=pd.read_csv(os.path.join(outdirforcsv, 'Extracted Jobs Info.csv'))
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    df=df.reset_index(drop=True)
    create_cluster(df)



if __name__ == '__main__':
      main()
      print('-----------Hirerarchical clustering of job data is complete. Check the csv file.-----------')
    