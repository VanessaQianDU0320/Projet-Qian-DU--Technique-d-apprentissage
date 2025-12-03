# %%
import numpy as np
import sys
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')


#Read Document split to X and Y 
def Read(input_file):
    
    df = pd.read_csv(input_file, sep = ",")
    X = df.drop("Class", axis=1).values
    Y = df["Class"].values
    return X, Y

#Split Document to Test and Train
from sklearn.model_selection import train_test_split
def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)
    return x_train,x_test,y_train,y_test

#Normaliser les donnees 
from sklearn.preprocessing import StandardScaler
def standard_scaler(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)
    return x_train_scaled, x_test_scaled

#MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
def minmax_scaler(x_train, X_test):
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(X_test)
    return x_train_scaled, x_test_scaled

#PCA
from sklearn.decomposition import PCA
def apply_pca(x_train_std, x_test_std, n_components=10, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    Xtp = pca.fit_transform(x_train_std)
    Xsp = pca.transform(x_test_std)
    return Xtp, Xsp, pca

def save_processed_data(data_dict, output_folder="data/processed"):

    os.makedirs(output_folder, exist_ok=True)

    for name, data in data_dict.items():
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data
            
        file_path = os.path.join(output_folder, f"{name}.csv")
        
        # Save 
        df.to_csv(file_path, index=False)

    print("Sauvegarde terminée avec succès !")




