import pandas as pd
from sklearn.preprocessing import FunctionTransformer
import numpy as np
def load_data(file):
    return pd.read_csv(file, header = 0)
def load_data_XY(file, selected_features = None):
    data = load_data(file)
    #scale ic50's using log
    transformer = FunctionTransformer(np.log10)
    Y = data.values[:, 1]
    Y = transformer.transform(Y).T 
    
    
    features = data.values[:, 5:]
    labels = data.columns.values[5:] 
    #remove all columns that never change
    filter_mask = np.where(np.all(features == features[0,:], axis = 0))
    X = np.delete(features, filter_mask, axis = 1)
    labels = np.delete(labels, filter_mask)
    
    if selected_features is not None: #filter out features
        X = X[:, selected_features]
        labels = labels[selected_features]
    strains = data.values[:, 3]
    return (X, Y.ravel(), labels, strains)