
import logging
from numpy.core.defchararray import isnumeric
from pyro.primitives import module, param
from sklearn.model_selection import KFold
import numpy as np
from hyperopt import fmin, tpe, hp

logger = logging.getLogger(__name__)

class Objective:

    types = dict(
        num_modules  = int,
        max_learning_rate = float,
        num_epochs = int,
        batch_size = int,
        num_layers = int,
        encoder_dropout = float,
        epochs_adj_range = int, 
        max_lr_scale = float,
        seed = int,
    )

    def __init__(self, estimator, X, cv = 5):

        self.estimator = estimator
        self.results = []
        self.best_score = np.inf
        if isinstance(cv, int):
            self.cv = KFold(cv)
        else:
            self.cv = cv
        self.X = X

    def _map_types(self, params):
        return {param : self.types[param](value) for param, value in params.items()}


    def __call__(self, params):

        params = self._map_types(params)

        #check previous attempts
        for run in self.results:
            if run['params'] == params:
                logging.info('Already tried this parameter. Skipping to next test.')
                return run['mean_score']

        logger.info('Proposed params: ' + str(params))
        self.estimator.set_params(**params)

        scores = []
        for train_idx, test_idx in self.cv.split(self.X):
            train_counts, test_counts = self.X[train_idx].copy(), self.X[test_idx].copy()

            try:
                self.estimator.fit(train_counts)
                scores.append(self.estimator.score(test_counts))
            except Exception as err:
                logger.error('Exception occured while training: ' + repr(err))
                scores.append(np.nan)
        
        mean_score = np.nanmean(scores)
        is_best = mean_score < self.best_score
        self.best_score = min(mean_score, self.best_score)
        logger.info('Scores: mean = {:.3e}'.format(mean_score) + ('*' if is_best else '') + ', trials = ' +\
            ', '.join(['{:.3e}'.format(score) for score in scores if np.isfinite(score)])
        )

        self.results.append(dict(
            params = params,
            scores = scores,
            mean_score = mean_score,
        ))

        return mean_score 

def minimize_objective(objective, space, max_evals, **kwargs):

    try:
        logger.info('Searching parameter space. Halt search with esc-I-I in jupyter or ctrl-c on terminal.')
        best = fmin(objective, space, algo=tpe.suggest, max_evals=100, verbose = False, **kwargs)
    except KeyboardInterrupt:
        logger.warn('Interruped parameter search.')
        best = objective.results[np.argmin([r['mean_score'] for r in objective.results])]['params']

    return best, objective.results


def nestle_model(estimator, X, cv = 5, max_evals = 100,
        module_adj_range = (-2, 2),
        batch_sizes = [32, 64, 128, 256],
        encoder_dropout = (0.01, 0.1),
        epochs_adj_range = [-10,10],
        **kwargs
    ):
        
    space = {
        'num_modules' : hp.uniform('num_modules', estimator.num_modules + module_adj_range[0], estimator.num_modules + module_adj_range[1]),
        'batch_size' : hp.choice('batch_size', batch_sizes),
        'seed' : hp.randint('seed', 2**32 - 1),
        'encoder_dropout' : hp.loguniform('encoder_dropout', np.log(encoder_dropout[0]), np.log(encoder_dropout[1])),
        'num_epochs' : hp.uniform('num_epochs', estimator.num_epochs + epochs_adj_range[0], estimator.num_epochs + epochs_adj_range[1]),
    }

    return minimize_objective(Objective(estimator, X, cv = cv), space, max_evals= max_evals, **kwargs)
    



def estimate_num_modules(estimator, X, module_range = (5, 55), cv = 5, max_evals = None, grid_evaluations = 8, 
    points_to_evaluate = None, **kwargs):
    
    space = {
        'num_modules' : hp.qloguniform('num_modules', np.log(module_range[0]), np.log(module_range[1]), 1),
    }

    if max_evals is None:
        max_evals = int(module_range[1] - module_range[0])

    if points_to_evaluate is None:
        points_to_evaluate = [
            {'num_modules' : int(i)} for i in 
            np.exp(np.linspace(np.log(module_range[0]), np.log(module_range[1]), grid_evaluations)) 
        ]


    return minimize_objective(Objective(estimator, X, cv = cv), space, max_evals= max_evals, 
        points_to_evaluate = points_to_evaluate, **kwargs)

    






