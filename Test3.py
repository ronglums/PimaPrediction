
#%% [markdown]
# # Predicting Diabetes
# ## Import Libraries
# 


#%%
import pandas as pd # pandas is a dataframe library
import matplotlib.pyplot as plt # matplotlib.pyplot plots data
import numpy as np # numpy provides N-dim object support
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB


#%% [markdown]
# ## Load and review data

df = pd.read_csv("D:/Work Docs/AI/Demos/Test3/data/pima-data.csv") # load Pima data
df.head(5)
def plot_corr(df, size=11):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.
    
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
        
    Displays:
        matrix of correlation between columns. Blue-cyan-yellow-red-darkred => less to more correlated
                                               0------------------------->1
                                               Expect a darkred line running from top to bottom right
    """
    
    corr = df.corr() # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr) # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns) # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns) # draw y tick marks

plot_corr(df)

del df['skin']

#%% [markdown]
# ## Check Data Types

diabetes_map = {True:1, False:0}
df['diabetes'] = df['diabetes'].map(diabetes_map)
df.head(5)

#%% [markdown]
# ## Spliting the data
# 70% for training, 30% for testing

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

x = df[feature_col_names].values # predictor feature columns (8 X m)
y = df[predicted_class_names].values # predicted class (1=true, 0=false) column (1 X m)
split_test_size = 0.30

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_test_size, random_state=42)
# test_size = 0.3 is 30%, 42 is the answer to everything

#%% [markdown]
# We check to ensure we have the desired 70% train, 30% test split of the data

print("{0:0.2f}% in training set".format((len(x_train)/len(df.index))*100))
print("{0:0.2f}% in test set".format((len(x_test)/len(df.index))*100))

#%% [markdown]
# ## Post-split Data Preparation

#%% [markdown]
# ### Impute with the mean
# Impute with mean all 0 readings
fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)

x_train = fill_0.fit_transform(x_train)
x_test = fill_0.fit_transform(x_test)

#%% [markdown]
# ## Training Initial Algorithm = Naive Bayes


# create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()

nb_model.fit(x_train, y_train.ravel())

#%% [markdown]
# ### Performance on Training data

# predict values using the training data
nb_predict_train = nb_model.predict(x_train)

# import the performance metrics library

# Accuracy
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))
print("")

#%% [markdown]
# ### Performance on Testing data

# predict values using the training data
nb_predict_test = nb_model.predict(x_test)

# import the performance metrics library
# from sklearn import metrics

# Accuracy
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))
print("")

#%% [markdown]
# ### Metrics

print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(y_test, nb_predict_test))

#%% [markdown]
# ## Retrain =  Random Forest 

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42) # Create random forest object

rf_model.fit(x_train, y_train.ravel())

#%% [markdown]
# ### Performance on Training data

# predict values using the training data
rf_predict_train = rf_model.predict(x_train)

# import the performance metrics library
from sklearn import metrics

# Accuracy
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))
print("")

#%% [markdown]
# ### Performance on Testing data

# predict values using the testing data
rf_predict_test = rf_model.predict(x_test)

# import the performance metrics library
from sklearn import metrics

# Accuracy
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))
print("")

#%% [markdown]
# ### Metrics

print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, rf_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(y_test, rf_predict_test))

#%% [markdown]
# ## Retrain = Logistic Regression

from sklearn.linear_model import LogisticRegression

lf_model = LogisticRegression(C=0.7, class_weight="balanced", random_state=42)
lf_model.fit(x_train, y_train.ravel())

#%% [markdown]
# ### Performance on Training data

# predict values using the training data
lf_predict_train = lf_model.predict(x_train)

# import the performance metrics library
from sklearn import metrics

# Accuracy
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_train, lf_predict_train)))
print("")

#%% [markdown]
# ### Performance on Testing data

# predict values using the training data
lf_predict_test = lf_model.predict(x_test)

# import the performance metrics library


# Accuracy
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_test, lf_predict_test)))
print("")

#%% [markdown]
# ### Metrics

print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, lf_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(y_test, lf_predict_test))

#%%
mdf = df.head(100)
insulin = mdf['insulin'] 
age = mdf['age']
bmi = mdf['bmi']

trace1 = go.Scatter3d(
    x=insulin,
    y=age,
    z=bmi,
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

#x2, y2, z2 = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 200).transpose()
x2 = mdf['glucose_conc'] 
y2 = mdf['thickness']
z2 = mdf['diastolic_bp']

trace2 = go.Scatter3d(
    x=x2,
    y=y2,
    z=z2,
    mode='markers',
    marker=dict(
        color='rgb(255, 127, 39)',
        size=12,
        symbol='circle',
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.9
    )
)
data = [trace1, trace2]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter')