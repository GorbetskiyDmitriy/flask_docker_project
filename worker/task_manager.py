#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from celery import Celery
from api_models import ML_models
from mongodb import db

CELERY_BROKER = os.environ["CELERY_BROKER"]
CELERY_BACKEND = os.environ["CELERY_BACKEND"]

celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)

mlmodels = ML_models()
mlmodels.make_backup()


@celery.task(name='mlmodels')
def task(method=None, **kwargs):
    if method == 'model_types':
        return mlmodels.get_available_model_types()

    elif method == 'create_model':
        return mlmodels.create_model(**kwargs)

    elif method == 'get_all_models':
        return mlmodels.models

    elif method == 'get_model':
        return mlmodels.get_model(**kwargs)

    elif method == 'update_model':
        errors = mlmodels.update_model(kwargs)
        if not errors:
            return 'Model updated'
        else:
            return errors

    elif method == 'delete_model':
        return mlmodels.delete(**kwargs)

    elif method == 'set_model_params':
        model_id = kwargs['model_id']
        del kwargs['model_id']
        return mlmodels.set_model_params(model_id, kwargs)

    elif method == 'get_available_opt_params':
        try:
            return mlmodels.get_available_opt_params(kwargs['model_type'])
        except Exception:
            return 'Incorrect dictionary passed|Dictionary should be passed'

    elif method == 'optimize_model_params':
        try:
            params, score = mlmodels.optimize_model_params(**kwargs)
        except Exception as e:
            print(e)
            return 'Wrong dictionary format for params'
        dic = {'params': params,
               'score': score}
        return dic

    elif method == 'fit':
        model_id = kwargs['model_id']
        del kwargs['model_id']
        res = mlmodels.fit(model_id, **kwargs)
        if isinstance(res, str):
            return res
        return 'Model fitted'

    elif method == 'predict':
        model_id = kwargs['model_id']
        del kwargs['model_id']
        preds = mlmodels.predict(model_id, **kwargs)
        return preds

    elif method == 'predict_proba':
        model_id = kwargs['model_id']
        del kwargs['model_id']
        preds_proba = mlmodels.predict_proba(model_id, **kwargs)
        return preds_proba

    elif method == 'get_scores':
        model_id = kwargs['model_id']
        del kwargs['model_id']
        scores = mlmodels.get_scores(model_id, **kwargs)
        return scores


@celery.task(name='add_row')
def add_row(**kwargs):
    db.insert_one(kwargs)
    return 'Row inserted'


@celery.task(name='delete_row')
def delete_row(model_id):
    db.delete_one({'_id': model_id})
    return 'Row deleted'


@celery.task(name='update_row')
def update_row(id_, **kwargs):
    db.find_one_and_replace({'_id': id_}, kwargs)
    return 'Row updated'


@celery.task(name='get_row')
def get_row(model_id):
    data = db.find({'_id': model_id})
    data = [d for d in data][0]
    try:
        if data['model'] != 'Not fitted':
            data['model'] = "Fitted"
    except Exception:
        return 'Wrong id'
    return data