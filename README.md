# Détection de fraude dans les sinistres d’assurance (Machine Learning)

Ce projet vise à **détecter automatiquement les sinistres frauduleux** à partir d’un dataset d’assurance.
Le notebook couvre tout le pipeline : **EDA → nettoyage → encodage → standardisation → SMOTE → entraînement de modèles → évaluation → validation croisée → ROC → SHAP**.

---

## Contenu du dépôt

- `detection de fraude.ipynb` : notebook principal (pipeline complet)
- `insurance_claims.csv` : dataset (à placer en local dans le dossier `data/`)

> ⚠️ Si ton repo est public, évite de pousser le dataset si tu n’as pas le droit (licence/confidentialité).

---

## Dataset

- Fichier : `insurance_claims.csv`
- Variable cible : `fraud_reported` (Y = fraude, N = non fraude)
- Données anonymisées

---

## Installation

### Option 1 : exécuter directement dans Jupyter
1. Installer Python (3.9+ recommandé)
2. Installer les dépendances :

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn shap jupyter
data/insurance_claims.csv
jupyter notebook
