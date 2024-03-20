import numpy as np

def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))

def calculate_distortion(X, centroids, labels):
    """
    Calculate the distortion metric for the given dataset and clustering.
    
    Parameters:
    - X: numpy array of shape (n_samples, n_features), the dataset.
    - centroids: numpy array of shape (n_clusters, n_features), the cluster centroids.
    - labels: numpy array of shape (n_samples,), the cluster labels for each sample.
    
    Returns:
    - distortion: float, the calculated distortion metric.
    """
    distortion = 0.0
    for i, point in enumerate(X):
        centroid = centroids[labels[i]]
        print(point.shape)
        print(centroid.shape)
        distortion += euclidean_distance(point, centroid) ** 2
    return distortion / len(X)

def calculate_purity(y, y_pred):
    """
    Calculate the purity score for the given true labels and predicted cluster labels.

    Parameters:
    - true_labels: A list of true labels for each document.
    - predicted_labels: A list of predicted cluster labels for each document.

    Returns:
    - The purity score as a float.
    """
    cm = metrics.cluster.contingency_matrix(y, y_pred)
    print(cm)
    # purity: note that we assume that we match the predicted label with the ground truth by `amax`
    correct_predictions = np.amax(cm, axis=0) # Assign each cluster to the class most frequent in the cluster
    score =  np.sum(correct_predictions) / np.sum(cm) 

    return score

def silhouette_score(X, labels):
    """
    Calculate the mean silhouette score of all samples.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), the dataset.
    - labels: numpy array of shape (n_samples,), the cluster labels for each sample.

    Returns:
    - silhouette score: float, mean silhouette score for all samples.
    """
    n_samples, _ = X.shape
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        # If there's only one cluster, silhouette score is not defined.
        return 0

    # Calculate pairwise distances between points
    distance_matrix = np.array([[euclidean_distance(x, y) for y in X] for x in X])

    # Initialize a and b arrays
    a = np.zeros(n_samples)
    b = np.zeros(n_samples)

    for i in range(n_samples):
        # Same cluster mask
        same_cluster = labels == labels[i]
        # Distance to points in the same cluster
        a[i] = np.mean(distance_matrix[i, same_cluster]) if np.sum(same_cluster) > 1 else 0
        
        # Distance to points in other clusters
        min_dist = np.inf
        for label in unique_labels:
            if label == labels[i]:
                continue
            other_cluster = labels == label
            dist = np.mean(distance_matrix[i, other_cluster])
            min_dist = min(min_dist, dist)
        b[i] = min_dist
    
    # Calculate silhouette scores for each sample
    s = (b - a) / np.maximum(a, b)

    # Return the mean silhouette score
    return np.mean(s)
    

