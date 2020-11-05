import os
import re
from math import sqrt
from timeit import default_timer as timer

from first_run import *
from numpy import quantile
from numpy import unique
from scipy.stats import entropy
from skmultiflow.utils import constants


# inputs data stream S, buffer size w

# W a queue mantains a sliding window with length w
# Let expert be a classifier
# Let D' be a subset of features
# D_baseline the baseline feature


class FeatureSelectionEvaluator(EvaluatePrequential):

    def __init__(self,
                 n_wait=200,
                 max_samples=100000,
                 batch_size=1,
                 pretrain_size=2000,
                 max_time=float("inf"),
                 metrics=None,
                 output_file=None,
                 show_plot=False,
                 restart_stream=True,
                 data_points_for_classification=False,
                 window_size=2000):  # categorical or numerical
        super().__init__(n_wait=n_wait,
                         max_samples=max_samples,
                         batch_size=batch_size,
                         pretrain_size=pretrain_size,
                         max_time=max_time,
                         metrics=metrics,
                         output_file=output_file,
                         show_plot=show_plot,
                         restart_stream=restart_stream,
                         data_points_for_classification=data_points_for_classification)

        self.window_length = window_size
        self.Wx = []  # window of x variables
        self.Wy = []  # window of y variables
        self.Dt = []  # set of selected features
        self.D_baseline = None  # baseline feature
        self.current_window_size = 0
        # self.type = type #is target regresion or classification
        self.features_built = False

    def get_info(self):
        info = self.__repr__()
        if self.output_file is not None:
            _, filename = os.path.split(self.output_file)
            info = re.sub(r"output_file=(.\S+),", "output_file='{}',".format(filename), info)

        return info

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fit all the models on the given data.
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.
        y: Array-like
            An array-like containing the classification labels / target values for all samples in X.
        classes: list
            Stores all the classes that may be encountered during the classification task. Not used for regressors.
        sample_weight: Array-like
            Samples weight. If not provided, uniform weights are assumed.
        Returns
        -------
        EvaluatePrequential
            self
        """
        # todo feature selection
        # add to windows
        # calculate entropy

        if self.model is not None:
            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION or \
                        self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.model[i].partial_fit(X=X, y=y, classes=classes, sample_weight=sample_weight)
                else:
                    self.model[i].partial_fit(X=X, y=y, sample_weight=sample_weight)
            return self
        else:
            return self

    def predict(self, X):
        """ Predicts with the estimator(s) being evaluated.
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.
        Returns
        -------
        list of numpy.ndarray
            Model(s) predictions
        """

        # todo set X to subset of features X[Dt]

        predictions = None
        if self.model is not None:
            predictions = []
            for i in range(self.n_models):
                predictions.append(self.model[i].predict(X))

        return predictions

    def evaluate(self, stream, model, model_names=None):
        """

        :param self:
        :param stream:
        :param model:
        :param model_names:
        :return:
        """

        # initialise
        # list of dicts, each dict counters and H(Di)
        # list of dicts, each dict counters of Y|Di, and H(Y|Di)
        # dict of class counters and H(Y)

        self.n_features = stream.n_features  #
        self.record = []
        cat_values = stream.cat_features_idx

        # todo tell nominal and numeric

        self._init_evaluation(model=model, stream=stream, model_names=model_names)
        if self._check_configuration():
            self._reset_globals()
            # Initialize metrics and outputs (plots, log files, ...)
            self._init_metrics()
            self._init_plot()
            self._init_file()

            self.Wx = []
            self.Dt = [i for i in range(0, self.n_features)]
            for i in range(0, self.n_features):
                if i in cat_values:
                    self.Wx.append(EntropyCounter(type="Categorical", length=self.window_length))
                else:
                    self.Wx.append(EntropyCounter(type="Numeric", length=self.window_length))
            if self._task_type == constants.CLASSIFICATION:
                self.Wy = EntropyCounter(type="Categorical", length=self.window_length)
                print("classy")
            else:
                self.Wy = EntropyCounter(type="Numeric", length=self.window_length)
                print("nummy")

            self.model = self._train_and_test()
            if self.show_plot:
                self.visualizer.hold()

            return self.model

    def _update_window(self, disc_all):
        # count entropies for each feature entropies
        for i in range(0, self.n_features + 1):
            # count and calculate entropy for H(Di)
            d = self.value_counters[i]
            value = disc_all[i]
            if not value in d.keys():
                d[value] = 0
            d[value] += 1
            d['count'] += 1
            self.enqueue_entropy(value, d)

        # count and calculate entropy for H(Y|Di)
        for i in range(0, self.n_features + 1):
            for j in range(0, self.n_features + 1):
                cond_counter = self.conditional_counters[i][j]
                value_i = disc_all[i]
                value_j = disc_all[j]
                # initialise if empty
                if not value_i in cond_counter.keys():
                    cond_counter[value_i] = {'entropy': 0, 'count': 0}
                if not value_j in cond_counter[value_i].keys():
                    cond_counter[value_i][value_j] = 0
                cond_counter[value_i][value_j] += 1
                cond_counter[value_i]['count'] += 1
                cond_counter['count'] += 1
                self.enqueue_entropy(value_j, cond_counter[value_i])

        # deque any values if we've overfilled
        if self.current_window_size > self.window_length:
            # set up discretized values
            disc_x = []
            for i in range(0, self.n_features):  # n_featuress_c
                disc_x.append(self.Wx[i].remove())
            disc_y = self.Wy.remove()
            disc_all = disc_x + [disc_y]

            self.current_window_size -= 1

            for i in range(0, self.n_features + 1):
                # count and calculate entropy for H(Di)
                d = self.value_counters[i]
                value = disc_all[i]
                d[value] -= 1
                d['count'] -= 1
                self.dequeue_entropy(value, d)

            for i in range(0, self.n_features + 1):
                for j in range(0, self.n_features + 1):
                    cond_counter = self.conditional_counters[i][j]
                    value_i = disc_all[i]
                    value_j = disc_all[j]

                    if not value_i in cond_counter.keys():
                        print("key not found: outer")
                    if not value_j in cond_counter[value_i].keys():
                        print("key not found: inner")

                    cond_counter[value_i][value_j] -= 1
                    cond_counter[value_i]['count'] -= 1
                    cond_counter['count'] -= 1
                    self.dequeue_entropy(value_j, cond_counter[value_i])

        # calculate entropies
        # for i in range(0, self.n_features + 1):
        #   # count and calculate entropy for H(Di)
        #    d = self.value_counters[i]
        #    self.entropy_calc(d)

        # count and calculate entropy for H(Y|Di)
        # for i in range(0, self.n_features + 1):
        #    for j in range(0, self.n_features + 1):
        #        cond_counter = self.conditional_counters[i][j]
        #        for key, item in cond_counter.items():
        #            if key != 'count' and key != 'entropy':
        #                self.entropy_calc(item)

    def update_window(self, x, y):
        """
        :param x:
        :param y:
        :return:
        """

        # set up discretized values
        self.current_window_size += 1
        disc_x = []
        for i in range(0, self.n_features):  # n_featuress_c
            disc_x.append(self.Wx[i].add(x[i]))
        disc_y = self.Wy.add(y)

        if self.current_window_size == self.window_length:
            self._rebuild_window()
        elif self.current_window_size > self.window_length:  # if windows set up
            disc_all = disc_x + [disc_y]
            self._update_window(disc_all)

    def _rebuild_window(self):
        # trigger windows to be rebuilt
        self.current_window_size = 0
        y = self.Wy.get()
        w = [[0 for j in range(0, self.window_length)] for i in range(0, self.n_features)]
        for i in range(0, self.n_features):  # n_featuress_c
            w[i] = self.Wx[i].get()
        w = asarray(w)

        self.value_counters = []  # Count values of each value for each variable [X1,...Xn,Y]
        for i in range(0, self.n_features + 1):
            self.value_counters.append({'entropy': 0, 'count': 0})

        self.conditional_counters = []  # Count conditional values of each value for each variable, matrix of dicts
        for i in range(0, self.n_features + 1):
            row = []
            # self.conditional_counters
            for j in range(0, self.n_features + 1):
                row.append({'count': 0})
            self.conditional_counters.append(row)

        # setup matrix, transpose or whaterver to get it into shape
        for i in range(0, self.window_length):
            disc_x = list(w[:, i])
            disc_y = [y[i]]

            disc_all = disc_x + disc_y
            self.current_window_size += 1
            self._update_window(disc_all)

    def SU(self, x, y):
        Hx = self.value_counters[x]['entropy']
        Hy = self.value_counters[y]['entropy']

        dict_outer = self.conditional_counters[x][y]
        count_outer = dict_outer['count']
        Hxy = 0  # note should ne H(Y|X) #todo verify this
        for key, item in dict_outer.items():
            if key != "entropy" and key != "count":
                Hxy += item['entropy'] * item['count'] / count_outer  # todo check this
        if Hx + Hy == 0:
            return 0
        SU1 = 2 * (Hy - Hxy) / (Hx + Hy)
        # print(SU1)
        return SU1

    def _train_and_test(self):
        """ Method to control the prequential evaluation.
        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers.
        Notes
        -----
        The classifier parameter should be an extension from the BaseClassifier. In
        the future, when BaseRegressor is created, it could be an extension from that
        class as well.
        """
        self._start_time = timer()
        self._end_time = timer()
        print('Feature Selection Prequential Evaluation')
        print('Evaluating {} target(s).'.format(self.stream.n_targets))

        actual_max_samples = self.stream.n_remaining_samples()
        if actual_max_samples == -1 or actual_max_samples > self.max_samples:
            actual_max_samples = self.max_samples

        first_run = True
        if self.pretrain_size > 0:
            print('Pre-training on {} sample(s).'.format(self.pretrain_size))

            X, y = self.stream.next_sample(self.pretrain_size)

            # todo feature selection
            X = self.feature_select_filter(X, y)  # filter features

            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION:
                    # Training time computation
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y, classes=self.stream.target_values)
                    self.running_time_measurements[i].compute_training_time_end()
                elif self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y, classes=unique(self.stream.target_values))
                    self.running_time_measurements[i].compute_training_time_end()
                else:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y)
                    self.running_time_measurements[i].compute_training_time_end()
                self.running_time_measurements[i].update_time_measurements(self.pretrain_size)
            self.global_sample_count += self.pretrain_size
            first_run = False

        update_count = 0
        print('Evaluating...')
        while ((self.global_sample_count < actual_max_samples) & (self._end_time - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
                X, y = self.stream.next_sample(self.batch_size)
                X = self.feature_select_filter(X, y)  # filter features

                if X is not None and y is not None:

                    # Test
                    prediction = [[] for _ in range(self.n_models)]
                    for i in range(self.n_models):
                        try:
                            # Testing time
                            self.running_time_measurements[i].compute_testing_time_begin()
                            prediction[i].extend(self.model[i].predict(X))
                            self.running_time_measurements[i].compute_testing_time_end()
                        except TypeError:
                            raise TypeError("Unexpected prediction value from {}"
                                            .format(type(self.model[i]).__name__))
                    self.global_sample_count += self.batch_size

                    for j in range(self.n_models):
                        for i in range(len(prediction[0])):
                            self.mean_eval_measurements[j].add_result(y[i], prediction[j][i])
                            self.current_eval_measurements[j].add_result(y[i], prediction[j][i])
                    self._check_progress(actual_max_samples)

                    # Train
                    if first_run:
                        for i in range(self.n_models):
                            if self._task_type != constants.REGRESSION and \
                                    self._task_type != constants.MULTI_TARGET_REGRESSION:
                                # Accounts for the moment of training beginning
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y, self.stream.target_values)
                                # Accounts the ending of training
                                self.running_time_measurements[i].compute_training_time_end()
                            else:
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y)
                                self.running_time_measurements[i].compute_training_time_end()

                            # Update total running time
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)
                        first_run = False
                    else:
                        for i in range(self.n_models):
                            self.running_time_measurements[i].compute_training_time_begin()
                            self.model[i].partial_fit(X, y)
                            self.running_time_measurements[i].compute_training_time_end()
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)

                    if ((self.global_sample_count % self.n_wait) == 0 or
                            (self.global_sample_count >= actual_max_samples) or
                            (self.global_sample_count / self.n_wait > update_count + 1)):
                        if prediction is not None:
                            self._update_metrics()
                        update_count += 1

                self._end_time = timer()
            except BaseException as exc:
                print("Exception", exc)
                # print("ahh")
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                    # break

        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        if len(set(self.metrics).difference({constants.DATA_POINTS})) > 0:
            self.evaluation_summary()
        else:
            print('Done')

        if self.restart_stream:
            self.stream.restart()

        print(self.record)

        return self.model

    def selectFeatures(self, n_max_attempts=3):
        """
        :param D: feature set
        :param nMaxAttempts: number of attempts for merit improvement
        :return: selected features Dt and baseline feature D_baseline
        """
        # sort D in descending order of SU(.,Y)
        DtOld = self.Dt.copy()
        D = [i for i in range(0, self.n_features)]  # all features
        D.sort(key=lambda x: self.SU(x, self.n_features), reverse=True)
        SUs = []
        for i in D:
            SUs.append(self.SU(i, self.n_features))
        # print(SUs)
        Dt = []  # selected at time t
        M = 0
        sRelevences = 0  # sum of relevances
        sRedundancies = 0  # sum of redundancies
        nAttempts = 1  # counter of attempts for merit improvement

        for Di in D:
            Dt.append(Di)
            relevance = self.SU(Di, self.n_features)
            sRelevences += relevance
            for D_j in Dt:
                if Di != D_j:  # todo check this
                    redundancy = self.SU(Di, D_j)
                    sRedundancies += redundancy
            n = len(Dt)
            # print(sRelevences,sRedundancies)
            merit = sRelevences / sqrt(n + 2 * sRedundancies)
            if merit > M:
                M = merit
                nAttempts = 1
            else:
                nAttempts += 1
                Dt.remove(Di)  # Dt \ Di
                # restore releveance and redundancy
                sRelevences -= self.SU(Di, self.n_features)
                for D_j in Dt:
                    if Di != D_j:
                        sRedundancies -= self.SU(Di, D_j)
            if nAttempts == n_max_attempts:
                break
        if (len(Dt) == 0):
            return [i for i in range(0, self.n_features)], 0, 0
        # print(Dt)
        baseline = Dt[len(Dt) - 1]
        baseline_SU = self.SU(baseline + 1, self.n_features)

        for d in D:
            if not d in Dt:
                maxRed = 0
                for di in Dt:
                    maxRed = max(maxRed, self.SU(d, di))
                SU = self.SU(d, self.n_features)
                if maxRed < SU and (SU > 0.005 or (d in DtOld)):
                    Dt.append(d)
                # elif d in DtOld and SU > maxRed/2:
                #    Dt.append(d)
                # if SU < baseline_SU:
                #    baseline_SU =SU
                #    baseline = d
                # pass
                # print("appending {}".format(d))
        Dt.sort()
        return (Dt, baseline, baseline_SU)  # use last feature as baseline

    def entropy_calc(self, dict):
        total = dict['count']
        if total == 0 or total == 1:
            dict['entropy'] = 0
        else:
            values = []
            for key, item in dict.items():
                if key != 'count' and key != 'entropy':
                    values.append(item / total)
            dict['entropy'] = entropy(values, base=2)

    def enqueue_entropy(self, value, dict):
        n = dict['count'] * 1.0
        n_i = dict[value] * 1.0
        h1 = dict['entropy']

        if n == 0:
            # print("zero values, en")
            h = 0
        elif n == 1 and (n_i == 0):
            h = 0
        elif n == 1 and (n_i == 1):
            h = 0
        elif n == 1:  # n_i > 1, doesn't exist
            h = 0
        elif n > 1 and n_i == 0:  # DOESN'T EXIST,
            h = h1
        elif n > 1 and n_i == 1:
            # h = (n-1)/n*(h1-log2((n-1)/n))
            self.entropy_calc(dict)
            h = dict['entropy']
        else:
            h = (n - 1.0) / n * (h1 - log2((n - 1.0) / n)) - n_i / n * log2(n_i / n) + \
                (n_i - 1.0) / n * log2((n_i - 1.0) / n)
        if h < 0:
            self.entropy_calc(dict)
            print("enq_ayes")
            h = dict['entropy']
        dict['entropy'] = h

    def dequeue_entropy(self, value, dict):
        n = dict['count'] * 1.0
        n_i = dict[value] * 1.0
        h1 = dict['entropy']

        if n < 0 or n_i < 0:
            print("error")
        elif n == 0 or n == 1:
            # print("zero values, de")
            h = 0
        elif n_i == 0:  # if cat reduced to zero
            self.entropy_calc(dict)
            h = dict['entropy']
        elif n_i == 1:
            # h = (n+1.0)/n*(h1 + 2.0/(n+1.0)*log2((2.0)/(n+1.0))- 1.0/(n+1.0)*log2(1.0/(n+1.0)))+log2(n/(n+1.0))
            self.entropy_calc(dict)
            h = dict['entropy']
        elif n_i == n:
            h = 0
        else:
            h = ((n + 1.0) / n) * \
                (h1 + (n_i + 1.0) / (n + 1.0) * log2((n_i + 1.0) / (n + 1.0)) - (n_i) / (n + 1.0) * log2(
                    n_i / (n + 1.0))) + \
                log2(n / (n + 1.0))
        if h < 0:
            self.entropy_calc(dict)
            h = dict['entropy']
            print("deq_ayes")
        dict['entropy'] = h

    def relearn(self):
        self._rebuild_window()

        y = self.Wy.window
        w = [[0 for j in range(0, self.window_length)] for i in range(0, self.n_features)]
        for i in range(0, self.n_features):  # n_featuress_c
            w[i] = self.Wx[i].window
        w = asarray(w).transpose()
        X = w[:, self.Dt]

        for i in range(self.n_models):
            self.model[i].reset()
            if self._task_type == constants.CLASSIFICATION:
                # Training time computation
                self.model[i].partial_fit(X=X, y=y, classes=self.stream.target_values)
            elif self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                self.model[i].partial_fit(X=X, y=y, classes=unique(self.stream.target_values))
            else:
                self.model[i].partial_fit(X=X, y=y)

        pass

    def feature_select_filter(self, x, y):
        """
        Top level container for code
        waits till window of size window_size is built,
        then calculates baseline features, selected feature set
        has a trigger to reset features, reset models
        :param x:
        :param y:
        :return:
        """
        with open("check_file.csv", "a+") as file:
            r, c = x.shape  # dimensions of x
            for i in range(0, r):
                xt = x[i]
                yt = y[i]
                self.update_window(xt, yt)  # update window,counters

                if self.features_built:
                    D = [i for i in range(0, self.n_features)]
                    # D.remove(self.D_baseline)
                    flag = False

                    SU_baseline = self.SU(self.D_baseline, self.n_features)
                    # print(SU_baseline,"baseline")
                    for Di in D:
                        SU_next = self.SU(Di, self.n_features)
                        if (SU_next > min(SU_baseline, self.SU_baseline) and (Di not in self.Dt)) or \
                                (SU_next < max(SU_baseline,
                                               self.SU_baseline) and Di in self.Dt):  # baseline feature is allowed to decrease in performance
                            flag = True  # so there a trade off between elite stable, and less elite unstable #and SU_next >0.04 )
                            break
                    # flag = True
                    if SU_baseline < 0.005 or SU_baseline < self.SU_baseline:
                        flag = True
                    if flag:
                        # file.write(str(self.global_sample_count))
                        # for n in range(0, self.n_features):
                        #    file.write(",")
                        #    su = self.SU(n,self.n_features)
                        #   file.write(str(su))
                        # print("SU")
                        # file.write("\n")

                        Dt_new, self.D_baseline, self.SU_baseline = self.selectFeatures()

                        if self.Dt != Dt_new:
                            self.Dt = Dt_new  # select new features and defines a new baseline features
                            self.record.append(
                                {'index': self.global_sample_count, 'features': self.Dt, 'base_SU': SU_baseline})
                            # learn a new model with instances in buffer given newly selected features in D
                            print(Dt_new, self.D_baseline)
                            SU_baseline = self.SU(self.D_baseline, self.n_features)
                            print(SU_baseline, "baseline")
                            # print(len(Dt_new), "Selected features")
                            self.relearn()  # each instances in buffer: #todo implement
                    # train model
                    else:
                        # return x[:,self.Dt]
                        pass
                        # train model, as per normal
                elif self.current_window_size == self.window_length:  # find the first time window lenght met
                    # condition met during first w instances obtained from S
                    # fselects first subset of relevent features iven instances stored in W and sets baselined D
                    # print("ayyeeee")
                    (self.Dt, self.D_baseline, self.SU_baseline) = self.selectFeatures()
                    SU_baseline = self.SU(self.D_baseline, self.n_features)
                    self.record.append({'index': self.global_sample_count, 'features': self.Dt, 'base_SU': SU_baseline})
                    print(self.Dt, self.D_baseline, "feature model built")
                    print(SU_baseline, "baseline")
                    self.relearn()
                    self.features_built = True
                    # return []
                else:  # histograms not build, return normal values (without filtering)
                    pass

            if self.D_baseline is None:
                return x
            else:
                filtered = x[:, self.Dt]
                return filtered


class EntropyCounter():
    """
    Class that keeps a window for each variable
    Encapsulates method for window
    """

    def __init__(self, type="Categorical", length=200, nBins=10):
        self.type = type
        self.length = length
        self.ready = False
        self.nBins = 10
        self.window = []
        self.must_reconstruct = False
        if type == "Categorical":
            pass
        else:
            self.sLayer = SecondLayer(nBins)  # 2st layer

    def rebuild(self):
        if self.type == "Categorical":
            pass
        else:
            self.sLayer.build_layer(self.window)

    def get(self):
        if self.type == "Categorical":
            return self.window
        else:  # numeric, must return
            return self.sLayer.window  # second layer holds history of added values

    def add(self, X):
        """
        X is a single value
        :param X:
        :return:
        """
        if self.length == len(self.window):
            self.ready = True

        if self.type == "Categorical":
            self.window.append(X)
            return X
        else:  # type is numeric
            value = self.discretize_add(X, self.nBins, self.length)
            return value

    def remove(self):
        if not self.ready:
            print("ahhh")

        if self.type == "Categorical":
            val = self.window.pop(0)
            return int(val)  # self.window.pop(0)
        else:
            val = self.discretize_remove(self.nBins, self.length)
            return val

    def discretize_remove(self, F, w):
        """
        Contract that only called after window is filled
        :param x:
        :param F:
        :param w:
        :return:
        """

        # remove from window
        to_remove = self.window.pop(0)  # V goes to V remove {xt}
        removed = self.sLayer.remove()  # remove
        return removed

        # if True:
        #    self.must_reconstruct = True
        # if self.must_reconstruct:
        #    self.sLayer.build_layer(self.window)

    def discretize_add(self, xt, F=10, w=200):
        """
        :param x:  stream
        :param F: number of partitions
        :param w: sliding window size
        :return: provide F equal width bins
        """
        # print(xt)
        self.must_reconstruct = False  # flag for second layer reconstruction

        self.window.append(xt)  # V goes to V union {xt}
        if len(self.window) > w:  # if size V > w:

            if True:  # self.sLayer.check_rebuild(xt):
                self.sLayer.build_layer(self.window)
            # increment with added
            added = self.sLayer.add(xt)
            return added

        elif len(self.window) == w:  # first window is complete, both layers are constructeud
            self.sLayer.build_layer(self.window)
        else:  # window not built yet, build
            pass


class SecondLayer:
    def __init__(self, numBins=10):
        self.numBins = numBins
        self.window = None
        self.layer = None
        self.quantiles = []
        self.max = None
        self.min = None

    def check_rebuild(self, X):
        if self.max == None or self.min == None:
            return True
        if (X > self.max or X < self.min):  # check bounds change
            return True
        if (max(self.layer) - min(self.layer)) / sum(self.layer) > 0.1:
            return True
        return False

    def build_layer(self, X):
        """
        Equal frequency histogram
        :param x:
        :return:
        """
        if self.layer is None:
            self.layer = [0 for i in range(0, self.numBins)]

        # setup quantile values, for 10 bins, want 11 values
        self.quantiles = []
        for i in range(0, self.numBins + 1):
            self.quantiles.append(quantile(X, i / self.numBins))
        self.max = self.quantiles[self.numBins]
        self.min = self.quantiles[0]

        # place each in quantile bin by finding first quantile marker its smaller than
        if self.window is None:  # only add if first run (so when rebuilding leave values as is)
            self.window = []
            for xi in X:
                self.add(xi)

        return self.layer

    def add(self, X):
        """
        Add to layer, returning count of updated bin
        :param X:
        :return:
        """
        for i in range(1, self.numBins + 1):
            if X <= self.quantiles[i]:
                self.layer[i - i] += 1  # add to matrix
                discretized_X = i - 1
                self.window.append(discretized_X)  # add
                return discretized_X  # return discretized value just added

        return self.numBins - 1

    def remove(self):
        """
        Remove from layer, returning value removed
        :param X:
        :return:
        """
        val = self.window.pop(0)  # access history of bin this belonged to
        self.layer[val] -= 1
        return val

    # def toCatX(self,x):
    #    for i in range(1, self.numBins+1):
    #        if x <= self.quantiles[i]:
    #            return i-1
    #    return self.numBins-1


def single_count_dict(dict):
    for key, item in dict.items():
        if key != "entropy" and key != "count":
            if item == dict["count"]:
                return True
    return False
