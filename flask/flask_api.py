#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from flask_restx import Api
from celery import Celery
from time import sleep
import os

CELERY_BROKER = os.environ["CELERY_BROKER"]
CELERY_BACKEND = os.environ["CELERY_BACKEND"]

celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)
app = Flask(__name__)
api = Api(app)


@app.route("/api/results", methods=['GET'])
def results():
    task_id = request.json['task_id']
    res = celery.AsyncResult(task_id)
    if res.state == "PENDING":
        return res.state
    else:
        try:
            return str(res.get())
        except Exception:
            try:
                return jsonify(res.get())
            except Exception:
                return res.get()


def results(task_id):
    res = celery.AsyncResult(task_id)
    sleep(0.5)
    if res.state == "PENDING":
        return res.state
    else:
        try:
            return str(res.get())
        except Exception:
            return res.get()


@app.route("/api/model_types", methods=['GET'])
def model_types():
    """
    Выводит доступные типы моделей для обучения
    """
    task = celery.send_task('mlmodels', args=['model_types'])
    task_id = task.id
    result = results(task_id)
    return f'Success, Task_id = {task_id}\n' + result, 200


@app.route("/api/create_model", methods=['POST'])
def create_model():
    """
    Создает модель. В качестве параметров надо передать:
    model_type,
    model_name (опционально),
    dataset_name (опционально)
    """
    try:
        request.json['model_type']
    except KeyError:
        task = celery.send_task('mlmodels', args=['model_types'])
        task_id = task.id
        return '''model_type should be passed  {}'''.format(results(task_id))
    task = celery.send_task('mlmodels', args=['create_model'], kwargs=request.json)
    task_id = task.id
    result = results(task_id)
    return f'Success, Task_id = {task_id}\n' + result, 200


@app.route("/api/get_model", methods=['GET'])
def get_all_models():
    """
    Выводит все модели
    """
    task = celery.send_task('mlmodels', args=['get_all_models'])
    task_id = task.id
    result = results(task_id)
    try:
        return f'Success, Task_id = {task_id}\n' + result, 200
    except TypeError:
        return result


@app.route("/api/get_model/<int:model_id>", methods=['GET'])
def get_model(model_id):
    """
    Выводит модель с указанным id
    """
    task = celery.send_task('mlmodels', args=['get_model'], kwargs={'model_id':model_id})
    task_id = task.id
    result = results(task_id)
    return f'Success, Task_id = {task_id}\n' + result, 200


@app.route("/api/update_model", methods=['POST'])
def update_model():
    """
    Обновление сведений о модели. Для обновления доступны:
    'model_name' - наименование модели,
    'dataset_name' - наименование датасета
    """
    task = celery.send_task('mlmodels', args=['update_model'], kwargs=request.json)
    task_id = task.id
    result = results(task_id)
    try:
        return f'Success, Task_id = {task_id}\n' + result, 200
    except TypeError:
        return result


@app.route("/api/delete_model/<int:model_id>", methods=['DELETE'])
def delete_model(model_id):
    """
    Удаление модели с указанным id
    """
    task = celery.send_task('mlmodels', args=['delete_model'], kwargs={'model_id':model_id})
    task_id = task.id
    result = results(task_id)
    return f'Success, Task_id = {task_id}\n' + result, 200


@app.route("/api/set_model_params/<int:model_id>", methods=['PUT'])
def set_model_params(model_id):
    """
    Установка указанных параметров для указанной модели
    """
    kwargs = request.json
    kwargs['model_id'] = model_id
    task = celery.send_task('mlmodels', args=['set_model_params'], kwargs=kwargs)
    task_id = task.id
    result = results(task_id)
    return f'Success, Task_id = {task_id}\n' + result, 200


@app.route("/api/available_opt_params", methods=['GET'])
def get_available_opt_params():
    """
    Возвращает доступные для оптимазации гиперпараметры.
    Необходимо передать тип модели.
    """
    try:
        request.json['model_type']
    except TypeError:
        pass
    task = celery.send_task('mlmodels', args=['get_available_opt_params'], kwargs=request.json)
    task_id = task.id
    result = results(task_id)
    return f'Success, Task_id = {task_id}\n' + result, 200


@app.route("/api/optimize_model_params/<int:model_id>", methods=['GET', 'PUT'])
def optimize_model_params(model_id):
    """
    Для модели подбираются наилучшие гиперпараметры в соответствии с доступным перебором.
    """
    kwargs = request.json
    kwargs['model_id'] = model_id
    task = celery.send_task('mlmodels', args=['optimize_model_params'], kwargs=kwargs)
    task_id = task.id
    return f'Success, Task_id = {task_id}', 200


@app.route("/api/fit/<int:model_id>", methods=['PUT'])
def fit(model_id):
    """
    Обучение модели
    """
    kwargs = request.json
    kwargs['model_id'] = model_id
    task = celery.send_task('mlmodels', args=['fit'], kwargs=kwargs)
    task_id = task.id
    return f'Success, Task_id = {task_id}', 200


@app.route("/api/predict/<int:model_id>", methods=['GET', 'PUT'])
def predict(model_id):
    """
    Предсказания модели
    """
    kwargs = request.json
    kwargs['model_id'] = model_id
    task = celery.send_task('mlmodels', args=['predict'], kwargs=kwargs)
    task_id = task.id
    return f'Success, Task_id = {task_id}', 200


@app.route("/api/predict_proba/<int:model_id>", methods=['GET', 'PUT'])
def predict_proba(model_id):
    """
    Предсказанные вероятности по классам
    """
    kwargs = request.json
    kwargs['model_id'] = model_id
    task = celery.send_task('mlmodels', args=['predict_proba'], kwargs=kwargs)
    task_id = task.id
    return f'Success, Task_id = {task_id}', 200


@app.route("/api/get_scores/<int:model_id>", methods=['GET', 'PUT'])
def get_scores(model_id):
    """
    Возвращаются посчитанные метрики качества
    """
    kwargs = request.json
    kwargs['model_id'] = model_id
    task = celery.send_task('mlmodels', args=['get_scores'], kwargs=kwargs)
    task_id = task.id
    return f'Success, Task_id = {task_id}', 200


# добавление и апдейты в бд делаются под капотом апи, поэтому крайне не рекомендую это
# делать самостоятельно, поскольку есть большая вероятность поломать все.
# Но все равно добавляю ручки для самостоятельного взаимодействия с бд.
@app.route("/api/add_row/<int:model_id>", methods=['PUT'])
def add_row(model_id):
    kwargs = request.json
    kwargs["_id"] = model_id
    task = celery.send_task('add_row', kwargs=kwargs)
    task_id = task.id
    result = results(task_id)
    return f'Success, Task_id = {task_id}\n' + result, 200


@app.route("/api/delete_row/<int:model_id>", methods=['DELETE'])
def delete_row(model_id):
    task = celery.send_task('delete_row', args=[model_id])
    task_id = task.id
    result = results(task_id)
    return f'Success, Task_id = {task_id}\n' + result, 200


@app.route("/api/update_row/<int:model_id>", methods=['POST'])
def update_row(model_id):
    model_params = request.json
    task = celery.send_task('update_row', args=[model_id], kwargs=model_params)
    task_id = task.id
    result = results(task_id)
    return f'Success, Task_id = {task_id}\n' + result, 200


@app.route("/api/get_row/<int:model_id>", methods=['GET'])
def get_row(model_id):
    task = celery.send_task('get_row', args=[model_id])
    task_id = task.id
    result = results(task_id)
    return f'Success, Task_id = {task_id}\n' + result, 200


if __name__ == '__main__':
    app.run(host = "0.0.0.0")
