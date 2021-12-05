#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask
from flask_restx import Api
from flask_pymongo import PyMongo
import os

login = os.environ["MONGO_LOG"]
password = os.environ["MONGO_PASS"]

app = Flask(__name__)
api = Api(app)
app.config["MONGO_URI"] = "mongodb+srv://{}:{}@cluster.7wf1r.mongodb.net/mlmodels?retryWrites=true&w=majority".format(
                           login, password)
mongo = PyMongo(app)
db = mongo.db.mlmodels
