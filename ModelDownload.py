#%%
# download model
#import azureml
import numpy as np
import json
#from azureml.core import Workspace, Run
import pandas as pd
from sklearn.externals import joblib  

# display the core SDK version number
#print("Azure ML SDK Version: ", azureml.core.VERSION)
#%%
#from azureml.core import Workspace
#from azureml.core.model import Model

""" ws = Workspace.get("DemoWorkspace", None, subscription_id='b856ff87-00d1-4205-af56-3af5435ae401')
model=Model(ws, 'test2')
model.download(target_dir = '.')
import os 
# verify the downloaded model file
os.stat('./pima-trained-model.pkl') """
#%%
X_test = pd.read_csv("D:/Work Docs/AI/Demos/PimaData/test-data.csv")
X_test
#%%
data_json = X_test.to_json(orient = "records")
print(data_json)

#%%
input_data = "{\"data\": " + data_json + "}" 
print(input_data)
pima_model = joblib.load("D:/Work Docs/AI/Demos/PimaData/pima-trained-model.pkl")

#%%
data=pd.read_json(input_data)
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
    print(result)

df = pd.DataFrame([num_preg,glucose_conc,diastolic_bp, thickness, insulin, bmi, diab_pred, age], index =['num_preg','glucose_conc','diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age'] ).T

#%%
pima_model.predict(df)