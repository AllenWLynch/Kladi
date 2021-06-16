import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy import stats
import logging
import optuna
from sklearn.utils.estimator_checks import parametrize_with_checks
    

class ModuleObjective:

    def __init__(self, estimator, X, cv = 5, 
        min_modules = 5, max_modules = 55, 
        min_epochs = 15, max_epochs = 50, 
        min_dropout = 0.01, max_dropout = 0.3,
        batch_sizes = [32,64,128], seed = 2556):

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

    def impute_performance(self, scores):

        num_folds = len(scores)

        features = np.array([
            scores[:num_folds] for pruned, scores in zip(self.was_pruned, self.fold_scores) if (not pruned and len(scores) >= num_folds)
        ]).reshape((-1,num_folds))

        labels = np.array(self.trial_scores)[~np.array(self.was_pruned)]

        impute_scores = np.array(scores)[np.newaxis, :]

        reg = LinearRegression().fit(features, labels)
        
        return max(reg.predict(impute_scores)[0], np.median(self.trial_scores))


    def __call__(self, trial):

        params = dict(
            num_modules = trial.suggest_int('num_modules', self.min_modules, self.max_modules, log=True),
            batch_size = trial.suggest_categorical('batch_size', self.batch_sizes),
            encoder_dropout = trial.suggest_float('encoder_dropout', self.min_dropout, self.max_dropout),
            num_epochs = trial.suggest_int('num_epochs', self.min_epochs, self.max_epochs, log = True),
            seed = np.random.randint(0, 2**32 - 1),
        )
        
        self.estimator.set_params(**params)

        cv_scores = []
        was_pruned = False
        for step, (train_idx, test_idx) in enumerate(self.cv.split(self.X)):

            train_counts, test_counts = self.X[train_idx].copy(), self.X[test_idx].copy()
            
            cv_scores.append(
                self.estimator.fit(train_counts).score(test_counts)
            )
            trial.report(np.mean(cv_scores), step)
                
            if trial.should_prune():
                logging.info('Trial pruned!')
                was_pruned = True
                break
        
        trial_score = self.impute_performance(cv_scores) if was_pruned else np.mean(cv_scores)
        self.was_pruned.append(was_pruned)
        self.fold_scores.append(cv_scores)
        self.trial_scores.append(trial_score)
        self.params.append(params)
        return trial_score

    def get_flat_results(self):
        
        if len(self.params) == 0:
            raise Exception('No trials have been attempted!')

        list_of_params = {param : [] for param in self.params[0].keys()}

        for trial in self.params:
            for param, value in trial.items():
                list_of_params[param].append(value)

        list_of_params = list_of_params.update(
            value = self.trial_scores,  
            was_pruned = self.was_pruned,
            fold_scores = np.array(self.fold_scores, dtype = object)
        )

        return list_of_params

