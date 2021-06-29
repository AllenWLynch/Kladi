import numpy as np
from pyro.primitives import param
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy import stats
import logging
import optuna
from sklearn.utils.estimator_checks import parametrize_with_checks
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModuleObjective:

    def __init__(self, estimator, X, cv = 5, 
        min_modules = 5, max_modules = 55, 
        min_epochs = 15, max_epochs = 50, 
        min_dropout = 0.01, max_dropout = 0.3,
        batch_sizes = [32,64,128], seed = 2556, 
        score_fn = None, prune_penalty = 0.01):

        self.params = []
        self.trial_scores = []
        self.fold_scores = []
        self.was_pruned = []

        if isinstance(cv, int):
            self.cv = KFold(cv, random_state = seed, shuffle= True)
        else:
            self.cv = cv
        self.estimator = estimator
        self.X = X

        self.min_modules = min_modules
        self.max_modules = max_modules
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.min_dropout = min_dropout
        self.max_dropout = max_dropout
        self.batch_sizes = batch_sizes
        self.score_fn = score_fn
        self.prune_penalty = prune_penalty

    
    def __call__(self, trial):

        params = dict(
            num_modules = trial.suggest_int('num_modules', self.min_modules, self.max_modules, log=True),
            batch_size = trial.suggest_categorical('batch_size', self.batch_sizes),
            encoder_dropout = trial.suggest_float('encoder_dropout', self.min_dropout, self.max_dropout),
            num_epochs = trial.suggest_int('num_epochs', self.min_epochs, self.max_epochs, log = True),
            seed = np.random.randint(0, 2**32 - 1),
        )

        
        self.estimator.set_params(**params)

        trial.set_user_attr('trial_params', params)
        trial.set_user_attr('completed', False)
        trial.set_user_attr('batches_trained', 0)
        cv_scores = []

        for step, (train_idx, test_idx) in enumerate(self.cv.split(self.X)):

            train_counts, test_counts = self.X[train_idx].copy(), self.X[test_idx].copy()
            
            self.estimator.fit(train_counts)
            if self.score_fn is None:
                cv_scores.append(self.estimator.score(test_counts))
            else:
                cv_scores.append(self.score_fn(self.estimator, test_counts))

            if step == 0:
                trial.report(1.0, 0)

            trial.report(np.mean(cv_scores) + (self.prune_penalty * 0.5**step), step+1)
                
            if trial.should_prune():
                trial.set_user_attr('batches_trained', step+1)
                raise optuna.TrialPruned()

        trial.set_user_attr('batches_trained', step+1)
        trial.set_user_attr('completed', True)
        trial_score = np.mean(cv_scores)

        return trial_score
