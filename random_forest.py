import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import mean_absolute_error, classification_report
from data_preprocessing import create_train_test_val_sets, read_processed_data

# Load data
x_mendeley, y_mendeley = read_processed_data("../data/processed/mendeley_processed.csv")
x_phiusiil, y_phiusiil = read_processed_data("../data/processed/phiusiil_processed.csv")

# Create splits
mendeley_sets = create_train_test_val_sets(x_mendeley, y_mendeley, label_col="Label", test_size=0.2, n_splits=5)
phiusiil_sets = create_train_test_val_sets(x_phiusiil, y_phiusiil, label_col="Label", test_size=0.2, n_splits=5)

# Hyperparameter tuning
def optimize_random_forest(X, y):
    params = {
        'n_estimators': [100, 300, 500, 750, 1000],
        'max_depth': [4, 5, 6, 8, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', None]  # handles class imbalance like scale_pos_weight in XGBoost
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42,n_jobs=-1)
    random_search = RandomizedSearchCV(rf, param_distributions=params, random_state=42, cv=skf.split(X, y))
    random_search.fit(X, y)

    print('\nBest hyperparameters:')
    print(random_search.best_params_)

    return random_search

print("Running hyperparameter tuning using Mendeley Dataset:")
rf_mendeley = optimize_random_forest(mendeley_sets["x_train_val"], mendeley_sets["y_train_val"])
print("Running hyperparameter tuning using PhiUSIIL Dataset:")
rf_phiusiil = optimize_random_forest(phiusiil_sets["x_train_val"], phiusiil_sets["y_train_val"])

# Train and evaluate (no feature selection)
def train_no_feature_selection(dataset, model):
    stratified_scores = []
    all_y_val = []
    all_y_pred = []

    for train_idx, val_idx in dataset["cv_splits"]:
        x_train = dataset["x_train_val"].iloc[train_idx]
        x_val = dataset["x_train_val"].iloc[val_idx]
        y_train = dataset["y_train_val"].iloc[train_idx]
        y_val = dataset["y_train_val"].iloc[val_idx]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        stratified_scores.append(mean_absolute_error(y_val, y_pred))

        all_y_val.extend(y_val)
        all_y_pred.extend(y_pred)

    print('Classification Report:')
    print(classification_report(all_y_val, all_y_pred))

print('Mendeley Results:')
train_no_feature_selection(mendeley_sets, rf_mendeley.best_estimator_)

print('PhiUSIIL Results:')
train_no_feature_selection(phiusiil_sets, rf_phiusiil.best_estimator_)
