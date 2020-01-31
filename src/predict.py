from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import joblib

def load_data(path):
    return joblib.load(path)

def predict(model, X_pca):
    return model.predict(X_pca)

if __name__ == '__main__':
    model = load_data('../models/model.pkl')
    X_pca = load_data('../models/X_pca.pkl')
    print(predict(model, X_pca))




