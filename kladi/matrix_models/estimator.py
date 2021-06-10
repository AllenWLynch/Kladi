from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator

class ModuleEstimator(BaseEstimator):

    def __init__(self,*, base_estimator, features, num_modules = 15, highly_variable = None, dropout = 0.2, hidden = 128, use_cuda = True,
        num_epochs = 200, batch_size = 32, min_learning_rate = 1e-6, max_learning_rate = 1, epochs_per_cycle = 2.5, triangle_decay = 0.97,
        tolerance = 1e-4):

        self.features = features
        self.num_modules = num_modules
        self.highly_variable = highly_variable
        self.dropout = dropout
        self.hidden = hidden
        self.use_cuda = use_cuda
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.epochs_per_cycle = epochs_per_cycle
        self.triangle_decay = triangle_decay
        self.tolerance = tolerance
        self.base_estimator = base_estimator

    def fit(self, X, training_bar = True):

        self.estimator = self.base_estimator(self.features, highly_variable = self.highly_variable, num_modules = int(self.num_modules), 
            dropout = self.dropout, hidden = self.hidden, use_cuda = self.use_cuda)
        self.estimator.training_bar = training_bar

        self.estimator.fit(X, num_epochs = self.num_epochs, batch_size = self.batch_size, min_learning_rate = self.min_learning_rate, 
            triangle_decay = self.triangle_decay, epochs_per_cycle = self.epochs_per_cycle, max_learning_rate = self.max_learning_rate, tolerance = self.tolerance)

        return self

    def score(self, X):
        return self.estimator.score(X)