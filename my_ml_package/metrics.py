from sklearn.model_selection import KFold, cross_val_score
import numpy as np

def variance_in_cv_scores(model, X, y):
    # Define a cross-validation strategy (e.g., 5-fold cross-validation)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and calculate scores for each fold
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

    # Calculate and return variability measures
    std_deviation = np.std(scores)
    return std_deviation

