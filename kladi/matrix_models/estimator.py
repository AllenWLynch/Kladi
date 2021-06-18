from sklearn.base import BaseEstimator
import numpy as np
from scipy import interpolate, sparse
import matplotlib.pyplot as plt
from kladi.matrix_models.model_selection import ModuleObjective
import optuna
import torch
import logging
from kladi.matrix_models.expression_model import ExpressionModel
from kladi.matrix_models.accessibility_model import AccessibilityModel
from gensim.matutils import Sparse2Corpus
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

class ExpressionTrainer(BaseEstimator):

    base_estimator = ExpressionModel

    def __init__(self,*, features, highly_variable = None, 
        num_modules = 15, encoder_dropout = 0.15, decoder_dropout = 0.2, hidden = 128, num_layers = 3,
        min_learning_rate = 1e-6, max_learning_rate = 1, num_epochs = 200, batch_size = 32, patience = 3, tolerance = 1e-4,
         use_cuda = True, seed = None): 

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
        self.tolerance = tolerance
        self.num_layers = num_layers
        self.patience = patience
        self.seed = seed

    def _get_score_fn(self, X):
        return None

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

    @staticmethod
    def _define_boundaries(learning_rate, loss, trim = [1,1]):

        assert(isinstance(learning_rate, np.ndarray))
        assert(isinstance(loss, np.ndarray))
        assert(learning_rate.shape == loss.shape)
        
        learning_rate = np.log(learning_rate)
        bounds = learning_rate.min()-1, learning_rate.max()+1
        
        x = np.concatenate([[bounds[0]], learning_rate, [bounds[1]]])
        y = np.concatenate([[loss.min()], loss, [loss.max()]])
        spline_fit = interpolate.splrep(x, y, k = 5, s= 5)
        
        x_fit = np.linspace(*bounds, 100)
        
        first_div = interpolate.splev(x_fit, spline_fit, der = 1)
        
        cross_points = np.concatenate([[0], np.argwhere(np.abs(np.diff(np.sign(first_div))) > 0)[:,0], [len(first_div) - 1]])
        longest_segment = np.argmax(np.diff(cross_points))
        
        left, right = cross_points[[longest_segment, longest_segment+1]]
        
        start, end = x_fit[[left, right]] + np.array([trim[0], -trim[1]])
        #optimal_lr = x_fit[left + first_div[left:right].argmin()]
        
        return np.exp(start), np.exp(end), spline_fit


    def tune_learning_rate_bounds(self, X, num_epochs = 3, eval_every = 20):

        self.gradient_lr, self.gradient_loss = self._make_estimator()._get_learning_rate_bounds(X, num_epochs = num_epochs, eval_every = eval_every,
            min_learning_rate = self.min_learning_rate, max_learning_rate = self.max_learning_rate)

        self.min_learning_rate, self.max_learning_rate, self.spline = self._define_boundaries(self.gradient_lr, self.gradient_loss, trim = [0,0.5])

        return self.min_learning_rate, self.max_learning_rate


    def set_learning_rates(self, min_lr, max_lr):
        self.set_params(min_learning_rate = min_lr, max_learning_rate= max_lr)


    def trim_learning_rate_bounds(self, min_trim = 0, max_trim = 0.5):

        try:
            self.gradient_lr
        except AttributeError:
            raise Exception('User must run "get_learning_rate_bounds" before running this function')

        assert(isinstance(min_trim, (int,float)) and min_trim >= 0)
        assert(isinstance(max_trim, (float, int)) and max_trim >= 0)

        self.min_learning_rate, self.max_learning_rate, self.spline = self._define_boundaries(self.gradient_lr, self.gradient_loss, trim = [min_trim, max_trim])
        return self.min_learning_rate, self.max_learning_rate


    def plot_learning_rate_bounds(self, figsize = (10,7), ax = None):

        try:
            self.gradient_lr
        except AttributeError:
            raise Exception('User must run "get_learning_rate_bounds" before running this function')

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize = figsize)

        x = np.log(self.gradient_lr)
        bounds = x.min(), x.max()

        x_fit = np.linspace(*bounds, 100)
        y_spline = interpolate.splev(x_fit, self.spline)

        ax.scatter(self.gradient_lr, self.gradient_loss, color = 'lightgrey', label = 'Batch Loss')
        ax.plot(np.exp(x_fit), y_spline, color = 'grey', label = '')
        ax.axvline(self.min_learning_rate, color = 'red', label = 'Min/Max Learning Rate')
        ax.axvline(self.max_learning_rate, color = 'red', label = '')

        legend_kwargs = dict(loc="upper left", markerscale = 1, frameon = False, fontsize='large', bbox_to_anchor=(1.0, 1.05))
        ax.legend(**legend_kwargs)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set(xlabel = 'Learning Rate', ylabel = 'Loss', xscale = 'log')
        return ax

    def tune_num_epochs(self,*, X, test_X, max_epochs = 200):
        
        test_lr = self.min_learning_rate + np.exp((np.log(self.min_learning_rate) - np.log(self.max_learning_rate))/2)

        num_epochs = self._make_estimator()._estimate_num_epochs(X = X, max_epochs = max_epochs, test_X = test_X, test_learning_rate = test_lr, 
            patience = self.patience, tolerance = self.tolerance, batch_size = self.batch_size)

        self.set_params(num_epochs = self.num_epochs)
        return num_epochs, num_epochs*2

    def tune_hyperparameters(self, X, iters = 200, cv = 5, min_modules = 5, max_modules = 55, 
        min_epochs = 20, max_epochs = 40, study = None):

        self.objective = ModuleObjective(self, X, cv = cv, min_modules = min_modules, max_modules = max_modules,
            min_epochs = min_epochs, max_epochs = max_epochs, score_fn = self._get_score_fn(X))

        if study is None:
            self.study = optuna.create_study(
                direction = 'minimize',
                pruner = optuna.pruners.MedianPruner(n_startup_trials = 5, 
                                                    n_warmup_steps = 0, interval_steps = 1, n_min_trials = 1)
            )
        
        try:
            self.study.optimize(self.objective, n_trials = iters)
        except KeyboardInterrupt:
            pass
        
        return self.study

    def get_tuning_results(self):

        try:
            self.objective
        except AttributeError:
            raise Exception('User must run "tune_hyperparameters" before running this function')

        return self.objective.get_flat_results()


    def get_best_params(self, top_n_models = 5):
        
        assert(isinstance(top_n_models, int) and top_n_models > 0)
        try:
            self.objective
        except AttributeError:
            raise Exception('User must run "tune_hyperparameters" before running this function')

        model_idx = np.argsort(self.objective.trial_scores)[:top_n_models]

        return [self.objective.params[i] for i in model_idx]


    def select_best_model(self, X, test_X, top_n_models = 5):

        scores = []
        best_params = self.get_best_params(top_n_models)
        for params in best_params:
            logging.info('Training model with parameters: ' + str(params))
            scores.append(self.set_params(**params).fit(X).score(test_X))
            logging.info('Score: {:.5e}'.format(scores[-1]))

        final_choice = best_params[np.argmin(scores)]
        logging.info('Set parameters to best combination: ' + str(final_choice))
        self.set_params(**final_choice)
        
        logging.info('Training model with all data.')

        if isinstance(X, np.ndarray):
            all_counts = np.vstack([X, test_X])
        else:
            all_counts = sparse.vstack([X, test_X])

        return self.fit(all_counts).estimator

    def score(self, X):
        return self.estimator.score(X)

    def save(self, filename):

        save_dict = dict(params = self.get_params(), model = self.estimator._get_save_data())
        torch.save(save_dict, filename)

    def load(self, filename):

        save_dict = torch.load(filename)
        self.set_params(**save_dict['params'])

        self._make_estimator()._load_save_data(save_dict['model'])

        return self.estimator


class CoherenceEvaluator:

    def __init__(self, estimator, X, topn = 1000):

        self.corpus = Sparse2Corpus(X, documents_columns = False)
        self.gensim_dict = Dictionary.from_corpus(self.corpus)
        self.topn = topn

    def __call__(self, estimator, X):

        topics = list(map(lambda x : list(x)[::-1], np.argsort(estimator.estimator._score_features(), -1)[:, -self.topn:].astype(str)))

        return -CoherenceModel(topics = topics, corpus= self.corpus, dictionary= self.gensim_dict, topn = self.topn, coherence='u_mass').get_coherence()


class AccessibilityTrainer(ExpressionTrainer):

    base_estimator = AccessibilityModel