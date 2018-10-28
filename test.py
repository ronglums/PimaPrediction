# download model
import azureml
from azureml.core import Workspace, Run

# display the core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

from azureml.core import Workspace
from azureml.core.model import Model

ws = Workspace.get("DemoWorkspace", None, subscription_id='b856ff87-00d1-4205-af56-3af5435ae401')
model=Model(ws, 'test1')
model.download(target_dir = '.')
import os 
# verify the downloaded model file
os.stat('./pima-trained-model.pkl')

