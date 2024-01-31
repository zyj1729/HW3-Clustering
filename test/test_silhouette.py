# write your silhouette score unit tests here
import pytest
import numpy as np
from sklearn.metrics import silhouette_samples
from cluster import Silhouette, make_clusters

def test_silhouette_score():
    X, y = make_clusters(n=300, m=2, k=3)
    silhouette = Silhouette()
    custom_scores = silhouette.score(X, y)
    sklearn_scores = silhouette_samples(X, y)

    # Convert custom_scores to a numpy array for comparison
    custom_scores = np.array(custom_scores)

    # Assert that the shape of the custom scores matches the sklearn scores
    assert custom_scores.shape == sklearn_scores.shape, "Shape of custom silhouette scores does not match sklearn's silhouette scores shape"

    # Compare the scores with a tolerance for numerical differences
    assert np.allclose(custom_scores, sklearn_scores, atol=1e-3), "Custom silhouette scores significantly differ from sklearn's silhouette scores"

