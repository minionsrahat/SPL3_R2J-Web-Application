from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from kneed import  KneeLocator

def calculate_optimal_cluster(model,X,range):
    clusters = range(2, 15)
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