import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # Ensure the number of labels matches the number of observations
        assert X.shape[0] == y.shape[0], "The number of labels should match the number of observations"

        result = []  # To store the silhouette score for each observation
        label_pool = np.array(list(set(y)))  # Unique set of cluster labels
        mids = []  # To store the mean (centroid) of each cluster

        # Calculate the centroid for each cluster
        for l in label_pool:
            cl = X[y == l]  # Extract all observations belonging to cluster l
            mid = cl.mean(axis=0)  # Calculate the mean for this cluster
            mids.append(mid)
        mids = np.array(mids)  # Convert list of centroids to a numpy array for further calculations

        # Calculate the silhouette score for each observation
        dist_mat = cdist(X, X)  # Pairwise distance matrix between all observations
        for i in range(X.shape[0]):
            x = X[i]  # The current observation
            same = dist_mat[i][y == y[i]]  # Distances from the current observation to all others in the same cluster
            a = sum(same) / (len(same) - 1)  # The average intra-cluster distance for the current observation

            inds = (label_pool != y[i])  # Indices of the nearest cluster other than its own
            pool = label_pool[inds]  # Labels of the nearest clusters
            mid_dist = cdist(mids[inds], [x])  # Distances from the current observation to the centroids of the nearest clusters
            closest = pool[mid_dist.argmin()]  # Label of the closest cluster
            
            diff = dist_mat[i][y == closest]  # Distances from the current observation to all observations in the closest cluster
            b = diff.mean()  # The average distance to the nearest cluster

            # Calculate the silhouette score for the current observation
            result.append((b - a) / max(a, b))

        return np.array(result)  # Return the silhouette scores as a numpy array