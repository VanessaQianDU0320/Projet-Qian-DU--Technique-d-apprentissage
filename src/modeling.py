# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# CART
def train_cart(x_train, y_train):
    model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    model.fit(x_train, y_train)
    print("FINISH CART")
    return model

# KNN
from imblearn.over_sampling import SMOTE

def apply_smote(x_train, y_train):
    sm = SMOTE(random_state=42)
    x_res, y_res = sm.fit_resample(x_train, y_train)
    return x_res, y_res

def train_knn(x_train, y_train, k=5):
    x_res, y_res = apply_smote(x_train, y_train)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_res, y_res)
    print("FINISH KNN")
    return model

# Random Forest
def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(x_train, y_train)
    print("FINISH RF")
    return model


import os
import joblib

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model(model, model_name, scaler_name):
    filename = f"{MODEL_DIR}/{model_name}_{scaler_name}.pkl"
    joblib.dump(model, filename)
    print(f"Model saved → {filename}")

