import requests
import json
import pandas as pd
import azureml
from azureml.core import Workspace
from azureml.core.webservice import Webservice

# load test data
X_test = pd.read_csv("./data/test-data.csv")
data_json = X_test.to_json(orient = "records")
input_data = "{\"data\": " + data_json + "}" 
print("input data:", input_data)

# send http request to scoring uri
scoring_uri = "http://40.78.43.160:80/score"
headers = {'Content-Type':'application/json'}
resp = requests.post(scoring_uri, input_data, headers=headers)
print("POST to url", scoring_uri)
print("prediction (0 means No, 1 means Yes):", resp.text) 