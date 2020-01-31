import argparse

def init_preprocess():
    parser = argparse.ArgumentParser(
            description='CLI for the segmentation project')
    parser.add_argument(
            'data_path', type=str, help='Path for the main file')
    parser.add_argument(
            'feat_path', type=str, help='Path to the feature file')
    parser.add_argument(
            'save_path', type=str, help='Path to save processed file')
    parser.add_argument(
            '--nrows', type=int, help='# rows to load')

    args = parser.parse_args()
    data_path = args.data_path
    feat_path = args.feat_path
    save_path = args.save_path
    nrows = args.nrows
    return (data_path, feat_path, save_path, nrows)

def init_train():
    parser = argparse.ArgumentParser(
            description='CLI for the training part')
    parser.add_argument(
            'data_path', type=str, help='Path for the main file')
    parser.add_argument(
            'save_model_path', type=str, help='Path to save the model file')
    parser.add_argument(
            'save_pca_path', type=str, help='Path to save the pca file')
    parser.add_argument(
            '--pca_components', type=int, help='Number of components')
    parser.add_argument(
            '--n_clusters', type=int, help='Number of clusters')

    args = parser.parse_args()
    data_path = args.data_path
    save_model_path = args.save_model_path
    save_pca_path = args.save_pca_path
    pca_components = args.pca_components or 116
    n_clusters = args.n_clusters or 19
    return (data_path,
            save_model_path,
            save_pca_path,
            pca_components,
            n_clusters)

def init_predict():
    parser = argparse.ArgumentParser(
            description='CLI for the prediction part')
    parser.add_argument(
            'model_path', type=str, help='Path for the model file')
    parser.add_argument(
            'pca_path', type=str, help='Path to the X_pca file')

    args = parser.parse_args()
    model_path = args.model_path
    pca_path = args.pca_path
    return (model_path, pca_path)
