# Détection de fraude par apprentissage supervisé  
Projet – Techniques d’Apprentissage Artificiel

Ce projet vise à analyser et comparer plusieurs modèles d’apprentissage supervisé appliqués à la détection de fraude sur des transactions de carte bancaire.  
Le travail inclut l’étude de trois modèles classiques — **CART**, **KNN**, **Random Forest** — évalués sous quatre stratégies de prétraitement, ainsi qu’une comparaison avec un **modèle CNN** développé dans un projet parallèle.

---

## 📂 Structure du projet
```text
project/
│
├── Data/
│ └── raw/
│  ├── creditcard.csv
│ └── processed/  -> Splited Data save area
│
├── models/  -> Model save area
│ ├── CART_MinMax.pkl
│ ├── CART_Original.pkl
│ ├── CART_PCA_10.pkl
│ ├── CART_Standard.pkl
│ ├── KNN_MinMax.pkl
│ ├── KNN_Original.pkl
│ ├── KNN_PCA_10.pkl
│ ├── KNN_Standard.pkl
│ ├── RF_MinMax.pkl
│ ├── RF_Original.pkl
│ ├── RF_PCA_10.pkl
│ └── RF_Standard.pkl
│
├── reports/
│ ├── figures/  -> Visualisation figures save area
│ ├── creditcard_analyse.xlsx
│ └── Rapport - Projet Techniques d’Apprentissage Artificiel.docx
│
├── src/
│ ├── evaluation.py
│ ├── modeling.py
│ └── prepare.py
├── main.py
├── requirements.txt
└── README.md
```

Les fichiers de modèles (.pkl), les figures, et les data splited ne sont pas inclus dans le dépôt GitHub
car ils dépassent la limite de taille de GitHub (>100 Mo).
Ils peuvent être régénérés en exécutant les scripts du dossier src/.

---

## 📊 Dataset

Le dataset provient de Kaggle :  
**Credit Card Fraud Detection**  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Caractéristiques principales :
- 284 807 transactions
- 30 variables numériques anonymisées (PCA)
- Classe fortement déséquilibrée :
  - Classe 0 : ~99,83 %
  - Classe 1 : ~0,17 % (fraude)

Les variables V1–V28 proviennent directement d’une PCA, limitant l’interprétabilité des caractéristiques.

---

## ⚙️ Prétraitement

Quatre versions du dataset ont été générées :

1. **Original** – aucune normalisation  
2. **StandardScaler** – centrage et réduction  
3. **MinMaxScaler** – mise à l’échelle [0, 1]  
4. **PCA_10** – réduction supplémentaire à 10 composantes  

Les transformations sont implémentées dans `src/prepare.py`.

---

## 🧠 Modèles implémentés

### Modèles classiques :
- CART (arbre de décision)
- KNN  
- Random Forest  

Chaque modèle est entraîné sur les 4 prétraitements → **12 modèles sauvegardés (.pkl)**.

### Modèle deep learning :
- **CNN** (comparaison uniquement)

---

## 📈 Métriques d’évaluation

Implémentées dans `src/evaluation.py` :

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC AUC  
- PR AUC  
- Matrice de confusion  
- Courbe ROC  
- Courbe Precision–Recall  

Étant donné le fort déséquilibre, le **Recall** et le **PR AUC** sont les métriques principales.

---

## 🚀 Exécution du projet

### 1. Installer les dépendances
pip install -r requirements.txt

### 2. Lancer le pipeline complet
python main.py

Ce script :
- charge les données  
- applique les prétraitements  
- entraîne les modèles  
- calcule les métriques  
- génère les figures dans `reports/figures`  
- sauvegarde les modèles dans `models/`

---

## 📑 Rapport & Figures

Le dossier `reports/` contient :
- Le rapport complet (DOCX)
- Les graphiques ROC/PR et matrices de confusion
- Analyse exploratoire (Excel)

---

## 🧾 Résultats principaux

- **Random Forest** : modèle le plus robuste et performant  
- **KNN** : très sensible à la normalisation ; bonne performance avec StandardScaler + SMOTE  
- **CART** : stable mais moins performant  
- **CNN** : meilleur rappel, mais moins explicable et plus coûteux en calcul

---

## 🔮 Améliorations possibles

- Ajouter des signaux métier (fréquence, montant inhabituel, géolocalisation…)  
- Utiliser les données brutes non PCA pour plus d’interprétabilité  
- Tester d'autre models 
- Ajouter une validation croisée  

---

## 👤 Auteur

**DU Qian**  
Master – Techniques d’Apprentissage Artificiel  
2025  

