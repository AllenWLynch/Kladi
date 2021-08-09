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
import gc
optuna.logging.set_verbosity(optuna.logging.WARNING)


try:
    from IPython.display import clear_output
    clear_output(wait=True)
    NOTEBOOK_MODE = True
except ImportError:
    NOTEBOOK_MODE = False

class ExpressionTrainer(BaseEstimator):

    base_estimator = ExpressionModel
    patience = 3
    tolerance = 1e-4

    @classmethod
    def load(cls, filename, **kwargs):

        save_dict = torch.load(filename)

        for k, v in kwargs.items():
            save_dict['params'][k] = v

        est = cls()
        est.set_params(**save_dict['params'])

        model = est._make_estimator()
        model._load_save_data(save_dict['model'])

        return model


    def __init__(self, features = None, highly_variable = None, 
        num_modules = 15, encoder_dropout = 0.15, decoder_dropout = 0.2, hidden = 128, num_layers = 3,
        min_learning_rate = 1e-6, max_learning_rate = 1, num_epochs = 200, batch_size = 32, beta = 0.95,
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
        self.num_layers = num_layers
        self.beta = beta
        self.seed = seed
        self.beta = beta

    def _get_score_fn(self, X):
        return None

    def _make_estimator(self):
        
        try:
            del self.estimator
            gc.collect()
            torch.cuda.empty_cache()
        except AttributeError:
            pass

        est = self.base_estimator(self.features, highly_variable = self.highly_variable, num_modules = self.num_modules,
            decoder_dropout = self.decoder_dropout, encoder_dropout = self.encoder_dropout, hidden = self.hidden, num_layers = self.num_layers,
            use_cuda = self.use_cuda, seed = self.seed)

        return est

    def _get_fit_params(self):
        return dict(
            batch_size = self.batch_size, beta = self.beta,
            min_learning_rate = self.min_learning_rate, max_learning_rate = self.max_learning_rate,
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


    def tune_learning_rate_bounds(self, X, num_epochs = 5, eval_every = 10):

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

    def _get_median_tuner(self):
        return optuna.pruners.MedianPruner(n_startup_trials = 5, 
                                                    n_warmup_steps = 0, interval_steps = 1, n_min_trials = 1)

    @staticmethod
    def _print_study(study, trial):

        def get_trial_desc(trial):

            if trial.user_attrs['completed']:
                return 'Trial #{:<3} | completed, score: {:.4e} | params: {}'.format(str(trial.number), trial.values[-1], str(trial.user_attrs['trial_params']))
            else:
                return 'Trial #{:<3} | pruned at step: {:<12} | params: {}'.format(str(trial.number), str(trial.user_attrs['batches_trained']), str(trial.user_attrs['trial_params']))

        if NOTEBOOK_MODE:
            clear_output(wait=True)
        else:
            print('------------------------------------------------------')

        try:
            print('Trials finished: {} | Best trial: {} | Best score: {:.4e}'.format(
            str(len(study.trials)),
            str(study.best_trial.number),
            study.best_value
        ), end = '\n\n')
        except ValueError:
            print('Trials finished {}'.format(str(len(study.trials))), end = '\n\n')        

        print('Modules | Trials (number is #folds tested)', end = '')

        study_results = sorted([
            (trial_.user_attrs['trial_params']['num_modules'], trial_.user_attrs['batches_trained'], trial_.number)
            for trial_ in study.trials
        ], key = lambda x : x[0])

        current_num_modules = 0
        for trial_result in study_results:

            if trial_result[0] > current_num_modules:
                current_num_modules = trial_result[0]
                print('\n{:>7} | '.format(str(current_num_modules)), end = '')

            print(str(trial_result[1]), end = ' ')
            #print(trial_result[1], ('*' trial_result[2] == study.best_trial.number else ''), end = '')
        
        print('', end = '\n\n')
        print('Trial Information:')
        for trial in study.trials:
            print(get_trial_desc(trial))


    def tune_hyperparameters(self, X, iters = 200, cv = 5, min_modules = 5, max_modules = 55, 
        min_epochs = 20, max_epochs = 40, batch_sizes = [32,64,128], study = None):

        self.objective = ModuleObjective(self, X, cv = cv, min_modules = min_modules, max_modules = max_modules,
            min_epochs = min_epochs, max_epochs = max_epochs, batch_sizes = batch_sizes, score_fn = self._get_score_fn(X))

        if study is None:
            self.study = optuna.create_study(
                direction = 'minimize',
                pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1.0, bootstrap_count=0, reduction_factor=3),
            )
        
        try:
            self.study.optimize(self.objective, n_trials = iters, callbacks = [self._print_study], 
            catch = (RuntimeError,ValueError),)
        except KeyboardInterrupt:
            pass
        
        return self.study

    def get_tuning_results(self):

        try:
            self.objective
        except AttributeError:
            raise Exception('User must run "tune_hyperparameters" before running this function')

        return self.study.trials


    def get_best_trials(self, top_n_models = 5):
        
        assert(isinstance(top_n_models, int) and top_n_models > 0)
        try:
            self.objective
        except AttributeError:
            raise Exception('User must run "tune_hyperparameters" before running this function')
        
        def score_trial(trial):
            if trial.user_attrs['completed']:
                try:
                    return trial.values[-1]
                except (TypeError, AttributeError):
                    pass

            return np.inf

        sorted_trials = sorted(self.study.trials, key = score_trial)

        return sorted_trials[:top_n_models]


    def get_best_params(self, top_n_models = 5):

        return [trial.user_attrs['trial_params'] for trial in self.get_best_trials(top_n_models)]


    def select_best_model(self, X, test_X, top_n_models = 5):

        scores = []
        best_params = self.get_best_params(top_n_models)
        for params in best_params:
            logging.info('Training model with parameters: ' + str(params))
            try:
                scores.append(self.set_params(**params).fit(X).score(test_X))
                logging.info('Score: {:.5e}'.format(scores[-1]))
            except RuntimeError as err:
                logging.error('Error occured while training, skipping model.')
                scores.append(np.inf)

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


'''class CoherenceEvaluator:

    def __init__(self, estimator, X, topn = 1000):

        self.corpus = Sparse2Corpus(X, documents_columns = False)
        self.gensim_dict = Dictionary.from_corpus(self.corpus)
        self.topn = topn

    def __call__(self, estimator, X):

        topics = list(map(lambda x : list(x)[::-1], np.argsort(estimator.estimator._score_features(), -1)[:, -self.topn:].astype(str)))

        return -CoherenceModel(topics = topics, corpus= self.corpus, dictionary= self.gensim_dict, topn = self.topn, coherence='u_mass').get_coherence()'''


class AccessibilityTrainer(ExpressionTrainer):

    base_estimator = AccessibilityModel
    tolerance = 1e-5