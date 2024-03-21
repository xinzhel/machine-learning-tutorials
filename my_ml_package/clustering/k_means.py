
import numpy as np
def initialize_centroids(X, k):
    """Randomly initialize k centroids from the dataset X."""
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def assign_points_to_centroids(X, centroids):
    """Assign each point in X to the nearest centroid."""
    assignments = []
    for point in X:
        distances = np.sqrt(((point - centroids) ** 2).sum(axis=1))
        closest_centroid = np.argmin(distances)
        assignments.append(closest_centroid)
    return np.array(assignments)

def update_centroids(X, assignments, k):
    """Update centroids to be the mean of points assigned to them."""
    new_centroids = np.array([X[assignments == i].mean(axis=0) for i in range(k)])
    return new_centroids

def k_means(X, k, iterations=100):
    """The K-means algorithm."""
    # Step 1: Initialization
    centroids = initialize_centroids(X, k)
    
    for _ in range(iterations):
        # Step 2: Assignment
        assignments = assign_points_to_centroids(X, centroids)
        
        # Step 3: Update
        new_centroids = update_centroids(X, assignments, k)
        
        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, assignments

def predict(X_new, centroids):
    """
    Assign new data points to the nearest centroid.
    
    Parameters:
    - X_new: numpy array of shape (n_samples, n_features), new data points.
    - centroids: numpy array of shape (k, n_features), the centroids.
    
    Returns:
    - assignments: numpy array of shape (n_samples,), cluster indices for each point.
    """
    assignments = []
    for point in X_new:
        distances = np.sqrt(((point - centroids) ** 2).sum(axis=1))
        closest_centroid = np.argmin(distances)
        assignments.append(closest_centroid)
    return np.array(assignments)

if __file__ == "__main__":
    # People data with Height cm, Weight kg
    X = np.array([
        [175, 75],  # Adult    1
        [60, 5],    # Baby     
        [50, 4],    # Baby     1
        [70, 7],    # Baby     
        [180, 80],  # Adult    1
        [178, 72],  # Adult (new) 1
        [172, 70],  # Adult (new)  
        [169, 74],  # Adult (new)  
        [55, 6],    # Baby (new)  
        [65, 8]     # Baby (new)  1
    ])

    # True labels
    y = ['A', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'B', 'B']

    centroids, _ = k_means(X, 2)
    predictions = predict(X, centroids)
    predictions