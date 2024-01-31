import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100, init_style: str = "rand"):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        assert type(k) == int and k > 0, "Invalid k, k should be a positive integer"
        assert type(tol) == float, "Invalid tol, tol should be a float"
        assert type(max_iter) == int, "Invalid max_iter, max_iter should be an integer"
        assert init_style in ["rand", "++"], "init_style can only be either rand for random initialization or ++ for kmeans++ initialization"
        # Initialize attributes
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.init_style = init_style

    def _init_plus(self, mat, k):
        """
        Initializes cluster centers using the k-means++ algorithm, which selects initial centers
        in a way that spreads them out within the dataset.
        """
        ind = np.random.choice(mat.shape[0], 1)[0]
        first = mat[ind]
        starts = [first]
        mat = np.concatenate((mat[:ind], mat[ind + 1:]))
        i = 0
        while i < k - 1:
            dist2 = cdist(mat, [first]) ** 2
            temp = dist2.sum()
            prob = dist2 / temp
            ind = np.random.choice(mat.shape[0], 1, p = prob.T[0])[0]
            first = mat[ind]
            mat = np.concatenate((mat[:ind], mat[ind + 1:]))
            starts.append(first)
            i += 1
        self.starts = starts
    
    def _sse(self, mat, centers):
        """
        Calculates the Sum of Squared Errors (SSE) for the dataset given the current cluster centers.
        Also returns the cluster assignments for each data point.
        """
        dist = cdist(mat, centers)
        seps = dist.argmin(axis = 1)
        result = [0 for i in range(len(centers))]
        for i in range(len(seps)):
            result[seps[i]] += dist[i][seps[i]] ** 2
        return seps, sum(result)
    
    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # Ensure the number of clusters does not exceed the number of data points
        assert self.k <= mat.shape[0], "k should be smaller than the number of data points"
        
        # Initialize cluster centers
        if self.init_style == "rand":
            temp = np.random.choice(mat.shape[0], self.k, replace=False)
            self.starts = mat[temp]
        elif self.init_style == "++":
            self._init_plus(mat, self.k)
        
        ii = 0
        centers = self.starts
        seps, ori = self._sse(mat, centers)
        _tol = self.tol + 1  # Initialize to a value larger than tol to enter the loop
        
        # Iteratively update cluster centers
        while ii < self.max_iter and _tol > self.tol:
            # Update each cluster center to be the mean of its assigned points
            for j in range(self.k):
                temp = mat[seps == j]
                if len(temp) > 0:
                    new = temp.mean(axis=0)
                    centers[j] = new
            seps, sse = self._sse(mat, centers)
            _tol = abs(sse - ori)  # Calculate change in error
            ori = sse  # Update previous error
            ii += 1
            
        # Store final labels, centers, and SSE
        self.labels = seps
        self.centers = centers
        self.num_cols = mat.shape[1]
        self.sse = sse
            
    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        # Ensure the data has the same number of features as the fitted data
        assert self.num_cols == mat.shape[1], "The number of features don't match"
        seps, _ = self._sse(mat, self.centers)
        return seps
        

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.sse

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centers
