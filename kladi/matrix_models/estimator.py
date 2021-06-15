from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator
from kladi.matrix_models.model_selection import estimate_num_modules

class ModuleEstimator(BaseEstimator):

    def __init__(self,*, base_estimator, features, highly_variable = None, 
        num_modules = 15, encoder_dropout = 0.15, decoder_dropout = 0.2, hidden = 128, num_layers = 3, 
        min_learning_rate = 1e-6, max_learning_rate = 1, num_epochs = 200, batch_size = 32, patience = 3, tolerance = 1e-4,
        policy = '1cycleLR', use_cuda = True, epochs_per_cycle = None, triangle_decay = None, seed = None): 

        self.features = features
        self.num_modules = num_modules
        self.highly_variable = highly_variable
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
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
        self.num_layers = num_layers
        self.policy = policy
        self.patience = patience
        self.seed = seed

    def _make_estimator(self):
        
        try:
            del self.estimator
        except AttributeError:
            pass

        est = self.base_estimator(self.features, highly_variable = self.highly_variable, num_modules = self.num_modules,
            decoder_dropout = self.decoder_dropout, encoder_dropout = self.encoder_dropout, hidden = self.hidden, num_layers = self.num_layers,
            use_cuda = self.use_cuda, seed = self.seed)

        return est

    def _get_fit_params(self):
        return dict(
            batch_size = self.batch_size, 
            min_learning_rate = self.min_learning_rate, max_learning_rate = self.max_learning_rate, 
            tolerance = self.tolerance, patience = self.patience
        )


    def fit(self, X, training_bar = False):
        
        self.estimator = self._make_estimator()

        self.estimator.training_bar = training_bar
        self.estimator.fit(X, num_epochs = self.num_epochs, **self._get_fit_params())

        return self

    def get_learning_rate_bounds(self, X, num_epochs = 3, eval_every = 20):

        lr, loss = self._make_estimator()._get_learning_rate_bounds(X, num_epochs = num_epochs, eval_every = eval_every,
            min_learning_rate = self.min_learning_rate, max_learning_rate = self.max_learning_rate)

        self._define_boundaries(lr, loss)

        return lr, loss

    def estimate_num_epochs(self,*, X, test_X, test_learning_rate, max_epochs = 200):
        
        num_epochs = self._make_estimator()._estimate_num_epochs(X = X, max_epochs = max_epochs, test_X = test_X, test_learning_rate = test_learning_rate, 
            patience = self.patience, tolerance = self.tolerance, batch_size = self.batch_size)

        self.set_params(num_epochs = self.num_epochs)
        return num_epochs

    def set_learning_rates(self, min_lr, max_lr):
        self.set_params(min_learning_rate = min_lr, max_learning_rate= max_lr)


    def estimate_num_modules(self, X, module_range = (5, 55), cv = 5, max_evals = None,
        grid_evaluations = 8, points_to_evaluate = None, **kwargs):

        best, results = estimate_num_modules(self, X, cv = cv, 
            module_range = module_range, max_evals = max_evals, grid_evaluations = grid_evaluations,
            points_to_evaluate = points_to_evaluate, **kwargs)

        self.set_params(**best)

        return best, results

    def score(self, X):
        return self.estimator.score(X)

    def save(self, filename):
        pass