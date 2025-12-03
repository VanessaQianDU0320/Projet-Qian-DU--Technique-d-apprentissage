# %%
import sys
import os
import importlib
import pandas as pd

CURRENT_DIR = os.getcwd()
sys.path.append(os.path.join(CURRENT_DIR, "src"))

from src import prepare,modeling,evaluation


if __name__ == "__main__":

    #Prepare data, split, normalisation, pca
    document_path = "data/raw/creditcard.csv"
    x,y = prepare.Read(document_path)
    x_train,x_test,y_train,y_test = prepare.split_data(x,y)

    #Modeling CART 
    #x origianl
    xtr_orig, xte_orig = x_train, x_test
    #x Standard
    xtr_std, xte_std = prepare.standard_scaler(x_train, x_test)
    #x MinMax
    xtr_mm,  xte_mm  = prepare.minmax_scaler(x_train, x_test)
    #x pca after Standard
    xtr_pca, xte_pca, _ = prepare.apply_pca(xtr_std, xte_std, n_components=10)

    data_to_save = {
    "X_train_orig": xtr_orig,
    "X_test_orig": xte_orig,
    "X_train_std": xtr_std,
    "X_test_std": xte_std,
    "X_train_mm": xtr_mm,
    "X_test_mm": xte_mm,
    "X_train_pca": xtr_pca,
    "X_test_pca": xte_pca
    }

    #Save split
    prepare.save_processed_data(data_to_save)
    
    scalers = {
        "Original": (xtr_orig, xte_orig),
        "Standard": (xtr_std,  xte_std),
        "MinMax":   (xtr_mm,   xte_mm),
        "PCA_10":   (xtr_pca,  xte_pca),
    }

    models = {
        "CART": modeling.train_cart,
        "KNN": modeling.train_knn,
        "RF": modeling.train_random_forest
    }

    results = []

    for scaler_name, (x_train_scaled, x_test_scaled) in scalers.items():
        print(f"\n=== Normalisation: {scaler_name} ===")

        for model_name, train_func in models.items():
            print(f"→ Training {model_name} with {scaler_name} data...")
            model = train_func(x_train_scaled, y_train)
            modeling.save_model(model, model_name, scaler_name)
            #Predict 
            y_pred = model.predict(x_test_scaled)
        
            y_proba = model.predict_proba(x_test_scaled)[:, 1] 
          
            metrics = evaluation.compute_metrics(y_test, y_pred, y_proba)
          
            results.append({
            "Scaler": scaler_name,
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "ROC_AUC": metrics["roc_auc"],
            "PR_AUC": metrics["pr_auc"]
            })
            
            evaluation.plot_roc_pr_curves(y_test, y_proba, model_name, scaler_name)
            evaluation.plot_confusion_matrix(y_test, y_pred, model_name, scaler_name)

    
    
    results_df = pd.DataFrame(results)
    output_file = "reports/creditcard_analyse.xlsx"
    results_df.to_excel(output_file, index=False)




    


    





