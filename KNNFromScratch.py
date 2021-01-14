import numpy as np

class KNNFromScratch():
    def __init__(self, k):
        self.k = k
    
    # Get training data
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X_test):
        dist = self.compute_dist(X_test) 
        return self.predict_label(dist)
    
    # Compute distance between each sample in X_test and X_train
    def compute_dist(self, X_test):
        test_size = X_test.shape[0]
        train_size = self.X_train.shape[0]
        dist = np.zeros((test_size, train_size))
        for i in range(test_size):
            for j in range(train_size):
                dist[i, j] = np.sqrt(np.sum((X_test[i,:] - self.X_train[j,:])**2))
        return dist
    
    # Return predicted label with given distance of X_test
    def predict_label(self, dist):
        test_size = dist.shape[0]
        y_pred = np.zeros(test_size)
        for i in range(test_size):
            y_indices = np.argsort(dist[i, :])
            k_closest = self.y_train[y_indices[: self.k]].astype(int)
            y_pred[i] = np.argmax(np.bincount(k_closest))
        return y_pred