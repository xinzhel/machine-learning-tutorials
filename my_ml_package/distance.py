import numpy as np

def euclidean_distance(v1, v2):
    """Compute the Euclidean distance between two numpy arrays."""
    return np.sqrt(np.sum((v1 - v2) ** 2))

def cosine_similarity(v1, v2):
    """Compute the cosine similarity between two numpy arrays."""
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    return dot_product / (magnitude_v1 * magnitude_v2)

def cosine_distance(v1, v2):
    """Compute the cosine distance between two numpy arrays."""
    return 1 - cosine_similarity(v1, v2)