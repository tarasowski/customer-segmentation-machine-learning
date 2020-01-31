import joblib
from  sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from cli import init_train
from functools import reduce

pipe = lambda fns: lambda x: reduce(lambda v, f: f(v), fns, x)

def data_load(path):
    return joblib.load(path)

def scaling(df):
    return StandardScaler().fit_transform(df)

def pca_transform(n_comp):
    def inner(X):
        return PCA(n_components=n_comp).fit_transform(X)
    return inner

def train(n_clusters):
    def inner(X_pca):
        model = KMeans(n_clusters, random_state=1).fit(X_pca)
        return (model, X_pca) 
    return inner

def save_status(model, path):
    joblib.dump(model, path)

main = lambda n_comp, n_clusters: pipe([
        scaling,
        pca_transform(n_comp),
        train(n_clusters),
        ])

if __name__ == '__main__':
  df = data_load('../models/df.pkl')
  model, X_pca = main(119, 19)(df)
  save_status(model, '../models/model.pkl')
  save_status(X_pca, '../models/X_pca.pkl')
