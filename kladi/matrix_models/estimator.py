from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator

class ModuleEstimator(BaseEstimator):

    def __init__(self,*, base_estimator, features, num_modules = 15, highly_variable = None, initial_counts = 15, dropout = 0.2, hidden = 128, use_cuda = True,
        num_epochs = 200, batch_size = 32, learning_rate = 0.1, lr_decay = 0.9, patience = 4,
        tolerance = 1e-4):

        self.features = features
        self.num_modules = num_modules
        self.highly_variable = highly_variable
        self.initial_counts = initial_counts
        self.dropout = dropout
        self.hidden = hidden
        self.use_cuda = use_cuda
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.patience = patience
        self.tolerance = tolerance
        self.base_estimator = base_estimator

    def fit(self, X):

        self.estimator = self.base_estimator(self.features, highly_variable = self.highly_variable, num_modules = self.num_modules, initial_counts = self.initial_counts, 
            dropout = self.dropout, hidden = self.hidden, use_cuda = self.use_cuda)
        self.estimator.training_bar = False

        self.estimator.fit(X, num_epochs = self.num_epochs, batch_size = self.batch_size, learning_rate = self.learning_rate, 
            lr_decay = self.lr_decay, patience = self.patience, tolerance = self.tolerance)

        return self

    def score(self, X):
        return self.estimator.score(X)