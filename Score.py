import json
import numpy as np
import os
import pickle
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from azureml.core.model import Model

def init():
    global model
    # retreive the path to the model file using the model name
    model_path = Model.get_model_path('test2')
    model = joblib.load(model_path)

def run(raw_data):
    data=pd.read_json(raw_data)
    num_preg,glucose_conc,diastolic_bp, thickness, insulin, bmi, diab_pred, age = [],[],[],[],[],[],[],[]
    for result in data['data']:
        num_preg.append(result[u'num_preg'])
        glucose_conc.append(result[u'glucose_conc'])
        diastolic_bp.append(result[u'diastolic_bp'])
        thickness.append(result[u'thickness'])
        insulin.append(result[u'insulin'])
        bmi.append(result[u'bmi'])
        diab_pred.append(result[u'diab_pred'])
        age.append(result[u'age'])
    df = pd.DataFrame([num_preg,glucose_conc,diastolic_bp, thickness, insulin, bmi, diab_pred, age], index =['num_preg','glucose_conc','diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age'] ).T

    # make prediction
    y_hat = model.predict(df)
    return json.dumps(y_hat.tolist())