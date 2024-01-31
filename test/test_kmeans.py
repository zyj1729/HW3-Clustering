# Write your k-means unit tests here
from cluster import make_clusters, KMeans
import pytest
import numpy as np
def test_kmeans_init():
    with pytest.raises(Exception) as excinfo:
        kmeans = KMeans(k=0)
    kmeans = KMeans(k=3)
    kmeans = KMeans(k=3, init_style = "++")
    assert kmeans.k == 3, "KMeans k parameter not initialized correctly"

def test_kmeans_fit_predict():
    mat, true_labels = make_clusters(n=300, m=2, k=3)
    kmeans = KMeans(k=3, tol=1e-4, max_iter=100)
    kmeans.fit(mat)
    predicted_labels = kmeans.predict(mat)

    assert predicted_labels.shape == true_labels.shape, "Predicted labels shape mismatch"

def test_kmeans_error():
    mat, _ = make_clusters(n=300, m=2, k=3)
    kmeans = KMeans(k=3)
    kmeans.fit(mat)
    error = kmeans.get_error()

    assert isinstance(error, float), "Error is not a float value"
    assert error >= 0, "Error is negative, which is incorrect"

def test_kmeans_centroids():
    mat, _ = make_clusters()
    kmeans = KMeans(k=3)
    kmeans.fit(mat)
    centroids = kmeans.get_centroids()

    assert isinstance(centroids, np.ndarray), "Centroids are not in an array format"
    assert centroids.shape == (3, mat.shape[1]), "Centroids shape is incorrect"

