import pandas as pd
from cli import init_preprocess, init_train, init_predict
from preprocess import main as preprocess, load_dfs
from train import main as train, save_status
from predict import predict
import joblib
import os


ENV = os.environ.get('ENV') 


def preprocess_step():
    data_path, feat_path, save_path, nrows = init_preprocess()
    dfs = load_dfs(data_path, feat_path, nrows)
    df = preprocess(dfs)
    joblib.dump(df, save_path)

def train_step():
    data_path, save_model_path, save_pca_path, n_comp, n_clusters = init_train()
    df = joblib.load(data_path)
    model, X_pca = train(n_comp, n_clusters)(df)
    save_status(model, save_model_path)
    save_status(X_pca, save_pca_path)

def predict_step():
    model_path, pca_path = init_predict()
    model = joblib.load(model_path)
    X_pca = joblib.load(pca_path)
    labels = predict(model, X_pca)
    return labels


if ENV == 'PREPROCESS':
    try:
        preprocess_step()
    except Exception as e:
        print(e)

elif ENV == 'TRAIN':
    try:
        train_step()
    except Exception as e:
        print(e)

elif ENV == 'PREDICT':
    try:
        labels = predict_step()
    except Exception as e:
        print(e)
    else:
        print(labels[:10])
