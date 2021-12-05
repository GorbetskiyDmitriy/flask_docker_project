#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from mongodb import db
from pymongo import DESCENDING

from utils import find_optuna_params, plot_AUC, get_metrics_scores


class ML_models:
    def __init__(self):
        """
        Класс на вход ничего не принимает, для этого есть
        встроенные методы. Однако, в init хранятся модели
        и счетчик для id.
        """
        self.models = []
        self.fitted_models = []
        try:
            self.counter = [a for a in db.find().sort('_id', direction=DESCENDING).limit(1)][0]['_id']
        except Exception:
            self.counter = 0

    @staticmethod
    def insert_data_in_db(model_dic):
        model_id = model_dic['model_id']
        try:
            db.delete_one({'_id': model_id})
        except Exception:
            pass
        model_dic_ = model_dic.copy()
        model_dic_['_id'] = model_id
        db.insert_one(model_dic_)

    @staticmethod
    def update_data_in_db(model_dic, model=None):
        model_id = model_dic['model_id']
        model_dic_ = model_dic.copy()
        model_dic_['_id'] = model_id
        if model is not None:
            model_dic_['model'] = pickle.dumps(model)
        else:
            pass
        db.delete_one({'_id': model_id})
        db.insert_one(model_dic_)

    def make_backup(self):
        try:
            models_info = [a for a in db.find().sort('_id')]
            for model_dic in models_info:
                model_holder = {'model_id': "",
                                'model_name': "",
                                'model_type': "",
                                'model_params': "",
                                'model': 'Not fitted',
                                'dataset_name': "",
                                'scores': {}}
                fitted_holder = {'model_id': model_dic['model_id'],
                                 'model': 'Not fitted'}
                for k in model_holder.keys():
                    if k in model_dic.keys():
                        if k == 'model' and model_dic[k] != 'Not fitted' \
                                and model_dic['model_type'] != "Boosting (LGBM)":
                            model_holder[k] = "Fitted"
                            fitted_holder[k] = pickle.loads(model_dic[k])
                        else:
                            model_holder[k] = model_dic[k]
                self.models.append(model_holder)
                self.fitted_models.append(fitted_holder)
            print('Back Up Succeed')
        except Exception:
            print('Back Up Failed')

    @staticmethod
    def get_available_model_types():
        """
        Метод выводит доступные для обучения типы моделей.
        Метод create_model принимает на вход
        наименования моделей, которые выводятся текущим методом.
        """
        available_types = ["LogisticRegression (LR)",
                           "Boosting (LGBM)"]
        return f'Available model types: {available_types}'

    def create_model(self, model_type, model_name='', dataset_name='', **kwargs):
        """
        Метод создает словарь с указанным типом модели.
        Доступные типы моделей можно узнать из метода get_available_model_types

        На выходе формируется словарь вида:
        {
         'model_id' - id модели,
         'model_name' - наименование модели (определяется
         пользователем),
         'model_type' -  тип модели (один из доступных),
         'model_params' - гиперпараметры модели (устанавливаются
         методом set_model_params),
         'model' - модель, заполняется при использовании метода fit,
         'dataset_name' - наименование датасета (определяется
         пользователем),
         'scores' - словарь с метриками для классификации
        }
        """
        self.counter += 1
        models_dic = {'model_id': self.counter,
                      'model_name': model_name,
                      'model_type': '',
                      'model_params': {'random_state': 42},
                      'model': 'Not fitted',
                      'dataset_name': dataset_name,
                      'scores': {}}
        fitted_dic = {'model_id': self.counter,
                      'model': 'Not fitted'}
        if model_type.lower() in "LogisticRegression (LR)".lower():
            models_dic['model_type'] = "LogisticRegression (LR)"
        elif model_type.lower() in "Boosting (LGBM)".lower():
            models_dic['model_type'] = "Boosting (LGBM)"
        else:
            self.counter -= 1
            return '''Wrong model type {} {}'''.format(model_type, self.get_available_model_types())
        self.models.append(models_dic)
        self.fitted_models.append(fitted_dic)
        self.insert_data_in_db(models_dic)
        return models_dic

    def get_model(self, model_id):
        """
        Метод выводит словарь с моделью с указанным id
        """
        for model in self.models:
            if model['model_id'] == model_id:
                return model
        return 'Model with id - {} doesnt exist'.format(model_id)

    def _get_fitted_model(self, model_id):
        """
        Метод выводи словарь с обученной моделью
        """
        for fit_model in self.fitted_models:
            if fit_model['model_id'] == model_id:
                return fit_model

    def update_model(self, model_dic={}):
        """
        Метод обновляет содержимое словаря модели.
        На вход поступает новый словарь, где ключи -
        ключи (только те, где надо обновление) исходного словаря,
        а значения - новые значения, которые необходимо обновить

        Для обновления доступны:
        'model_name' - наименование модели,
        'dataset_name' - наименование датасета
        """
        try:
            model = self.get_model(model_dic['model_id'])
            if isinstance(model, str):
                return model
        except KeyError:
            return 'Incorrect dictionary passed.'
        except TypeError:
            return 'Dictionary should be passed.'
        errors = []
        not_changeable = []
        for k, v in model_dic.items():
            if k in ['model_name', 'dataset_name']:
                model[k] = v
            elif k == 'model_id':
                continue
            elif k == 'model_params':
                errors.append("model_params is allowed to change only via set_model_params()")
            else:
                not_changeable.append(k)
        errors.append(f'Model attributes: {not_changeable} is not allowed to change')
        self.update_data_in_db(model)
        return errors

    def delete(self, model_id):
        """
        Метод удаляет модель с указанным id.
        """
        model = self.get_model(model_id)
        if not isinstance(model, dict):
            return model
        fitted_model = self._get_fitted_model(model_id)
        self.fitted_models.remove(fitted_model)
        self.models.remove(model)
        db.delete_one({'_id': model_id})
        return "Model has been deleted"

    def set_model_params(self, model_id, params={}):
        """
        ВАЖНО!
        Метод устанавливает гиперпараметры для модели.
        Для модели следует подавать только те гиперпараметры,
        которые доступны для этой модели.

        Можно было здесь сделать много проверок. Но считаю,
        что гибкость в данном случае важнее.
        """
        try:
            model_dic = self.get_model(model_id)
            if isinstance(model_dic, str):
                return model_dic
        except KeyError:
            return 'Incorrect dictionary passed.'
        except TypeError:
            return 'Dictionary should be passed.'
        if params is None:
            params = model_dic['model_params']
        try:
            if model_dic['model_type'] == 'LogisticRegression (LR)':
                LogisticRegression(**params)
            else:
                LGBMClassifier(**params)
        except TypeError:
            return 'Wrong params for model'
        fit_model = self._get_fitted_model(model_id)
        if model_dic['model_params'] != params:
            model_dic['model'] = 'Not fitted'
            model_dic['scores'] = {}
            fit_model['model'] = 'Not fitted'
        model_dic['model_params'] = params
        self.update_data_in_db(model_dic)
        return model_dic

    def get_available_opt_params(self, model_type, **kwargs):
        """
        Метод выводит доступные для оптимизации гиперпараметры
        для определенного типа модели.
        """
        if model_type.lower() in "LogisticRegression (LR)".lower():
            return ['C', 'tol', 'max_iter']
        elif model_type.lower() in "Boosting (LGBM)".lower():
            return ['boosting_type', 'n_estimators', 'learning_rate',
                    'num_leaves', 'max_depth', 'min_child_weight',
                    'reg_alpha', 'reg_lambda']
        else:
            return '''Wrong model type {} {}'''.format(model_type, self.get_available_model_types())

    @staticmethod
    def _transform_data(X=None, y=None):
        if X is None:
            y = pd.DataFrame(y).sort_index().values.flatten()
            return y
        elif y is None:
            X = pd.DataFrame(X).sort_index().values
            return X
        else:
            X = pd.DataFrame(X).sort_index().values
            y = pd.DataFrame(y).sort_index().values.flatten()
            return X, y

    def optimize_model_params(self, model_id, X, y, params_to_optimaze,
                              metric='auc_roc', max_trails=100, max_time=120, **kwargs):
        """
        ВАЖНО!
        Метод подбирает гиперпараметры для модели.
        Для модели следует подавать только те гиперпараметры,
        которые доступны для этой модели.

        Можно было здесь сделать много проверок. Но считаю,
        что гибкость в данном случае важнее.

        Доступные для перебора гиперпараметры можно посмотреть
        в методе get_available_opt_params.

        В словаре указывается наименование гиперпараметра
        и в качестве значения передается list c двумя значениями,
        где первое значение минимальное значение для перебора,
        а второе соответственно - максимальное.
        Для категориальных признаков следует указать все
        необходимые для перебора гиперпараметры

        Пример словаря с гиперпараметрами:
        params_to_optimaze = {
        'boosting_type': ['goss', 'gbdt', 'dart'],
        'n_estimators': [322, 1488],
        'learning_rate': [0.1, 0.2],
        'num_leaves': [10, 100],
        'max_depth': [1, 10],
        'min_child_weight': [2, 100],
        'reg_alpha': [0.1, 0.2],
        'reg_lambda': [0.1, 0.2]
        }
        """
        print(params_to_optimaze)
        X, y = self._transform_data(X, y)

        if np.unique(y).shape[0] != 2 and 'auc' in metric.lower():
            metric = 'f1'
        model_dic = self.get_model(model_id)
        available_params = self.get_available_opt_params(model_dic['model_type'])

        if model_dic['model_type'] == "Boosting (LGBM)":
            m = LGBMClassifier()
            params = m.get_params()
            for k, v in params.items():
                params[k] = [v, v]
        else:
            params = {
                'C': [1, 1],
                'tol': [0.0001, 0.0001],
                'max_iter': [100, 100]
            }
        for k, v in params_to_optimaze.items():
            if k not in available_params:
                continue
            elif k == 'boosting_type':
                params[k] = v
            elif len(v) == 2:
                params[k] = v
            else:
                return 'Wrong dictionary format for param {}'.format(k)

        best_params, best_score = find_optuna_params(X, y, model_dic['model_type'],
                                                     params, metric=metric,
                                                     max_trails=max_trails, max_time=max_time)

        prev_params = model_dic['model_params']

        for k, v in best_params.items():
            prev_params[k] = v
        model_dic['model_params'] = prev_params
        self.update_data_in_db(model_dic, model='Not fitted')
        return best_params, best_score

    def fit(self, model_id, X, y, **kwargs):
        """
        Обучение и сохранение указанной модели в словарь.
        """
        X, y = self._transform_data(X, y)
        model_dic = self.get_model(model_id)
        fitted_model = self._get_fitted_model(model_id)
        if model_dic['model_type'] == "LogisticRegression (LR)":
            model = LogisticRegression(**model_dic['model_params'])
        elif model_dic['model_type'] == "Boosting (LGBM)":
            model = LGBMClassifier(**model_dic['model_params'])
        try:
            model.fit(X, y)
        except ValueError as e:
            return f'{e}'
        model_dic['model'] = 'Fitted'
        fitted_model['model'] = model
        self.update_data_in_db(model_dic, model=model)
        return model_dic

    def predict(self, model_id, X, json=True, **kwargs):
        """
        Метод для формирования предсказаний.
        Аналогичен тому, который используется в scikit-learn
        """
        X = self._transform_data(X)
        # Просто для проверки id модели
        model_dic = self.get_model(model_id)
        fitted_model = self._get_fitted_model(model_id)
        model = fitted_model['model']
        try:
            preds = model.predict(X)
        except AttributeError:
            return 'Model not fitted yet, fit first'
        if json:
            return pd.DataFrame(preds).to_json()
        else:
            return preds

    def predict_proba(self, model_id, X, proba=False, json=True, **kwargs):
        """
        Метод для формирования вероятности предсказаний.
        """
        X = self._transform_data(X)
        # Просто для проверки id модели
        model_dic = self.get_model(model_id)
        fitted_model = self._get_fitted_model(model_id)
        model = fitted_model['model']
        n_classes = model.classes_.shape[0]
        if proba or n_classes != 2:
            try:
                preds_proba = model.predict_proba(X)
            except AttributeError:
                return 'Model not fitted yet, fit first'
        else:
            try:
                preds_proba = model.predict_proba(X)[:, 1]
            except AttributeError:
                return 'Model not fitted yet, fit first'
        if json:
            return pd.DataFrame(preds_proba).to_json()
        else:
            return preds_proba

    def get_scores(self, model_id, y_true, y_predicted=None, X=None, **kwargs):
        """
        Метод подсчитывает метрики качества классификации.
        В метод можно передать как сформированные уже предсказания,
        так и просто выборку для которой надо сформировать предсказания.

        Если используется бинарная классификация, то дополнительно
        считаются ROC_PR и ROC_AUC
        """
        model_dic = self.get_model(model_id)
        y_true = self._transform_data(y=y_true)
        if y_predicted is None and X is None:
            return 'At least y_predicted or X should be filled'
        if np.unique(y_true).shape[0] == 2:
            if y_predicted is None:
                y_predicted = self.predict_proba(model_id, X, json=False)
            else:
                y_predicted = self._transform_data(y=y_predicted)
            y_predicted_0_1 = np.where(y_predicted > 0.5, 1, 0)
            metrics = get_metrics_scores(y_true, y_predicted_0_1)
            metrics['ROC_PR'], metrics['ROC_AUC'] = \
                plot_AUC(y_true, y_predicted)
        else:
            if y_predicted is None:
                y_predicted = self.predict(model_id, X, json=False)
            else:
                y_predicted = self._transform_data(y=y_predicted)
            metrics = get_metrics_scores(y_true, y_predicted)
        model_dic['scores'] = metrics
        model = self._get_fitted_model(model_id)
        self.update_data_in_db(model_dic, model=model)
        return model_dic
