# Predicting Diabetes
# Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib  

DATA_PATH = "D:/Work Docs/AI/Demos/PimaData/pima-data.csv"
BEST_SCORE_C_VAL = 0.3
RANDOM_STATE = 42
MODEL_PATH = "D:/Work Docs/AI/Demos/Test6/data/pima-trained-model.pkl"

def load_data(DATA_PATH):
    """Load data"""
    return pd.read_csv(DATA_PATH) # load Pima data

def cleanup_data(df):
    """Clean up data"""
    del df['skin']
    diabetes_map = {True:1, False:0}
    df['diabetes'] = df['diabetes'].map(diabetes_map)
    return df

def split_data(df):
    """Spliting the data
    # 70% for training, 30% for testing"""
    feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
    predicted_class_names = ['diabetes']
    x = df[feature_col_names].values # predictor feature columns (8 X m)
    y = df[predicted_class_names].values # predicted class (1=true, 0=false) column (1 X m)
    split_test_size = 0.30
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_test_size, random_state=42)
    # test_size = 0.3 is 30%, 42 is the answer to everything
    # We check to ensure we have the desired 70% train, 30% test split of the data
    print("{0:0.2f}% in training set".format((len(x_train)/len(df.index))*100))
    print("{0:0.2f}% in test set".format((len(x_test)/len(df.index))*100))
    return x_train, x_test, y_train, y_test

def post_split_data_cleanup(x_train, x_test):
    """Post-split Data Preparation
    Impute with the mean"""
    # Impute with mean all 0 readings
    fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)
    x_train = fill_0.fit_transform(x_train)
    x_test = fill_0.fit_transform(x_test)
    return x_train, x_test

def train_with_naive_bayes(x_train, x_test, y_train, y_test):
    """ Training Initial Algorithm = Naive Bayes
    # create Gaussian Naive Bayes model object and train it with the data"""
    nb_model = GaussianNB()
    nb_model.fit(x_train, y_train.ravel())
    # ### Performance on Training data
    # predict values using the training data
    nb_predict_train = nb_model.predict(x_train)
    # Metrics
    # ### Performance on Training data
    print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))
    print("")
    # ### Performance on Testing data
    # predict values using the training data
    nb_predict_test = nb_model.predict(x_test)
    print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))
    print("")
    print("Confusion Matrix")
    print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test)))
    print("")
    print("Classification Report")
    print(metrics.classification_report(y_test, nb_predict_test))

def train_with_random_forest(x_train, x_test, y_train, y_test, random_state):
    """Retrain =  Random Forest"""
    rf_model = RandomForestClassifier(random_state=42) # Create random forest object
    rf_model.fit(x_train, y_train.ravel())
    # Performance on Training data
    # Predict values using the training data
    rf_predict_train = rf_model.predict(x_train)
    # Metrics
    print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))
    print("")
    # Performance on Testing data
    # Predict values using the testing data
    rf_predict_test = rf_model.predict(x_test)
    # Accuracy
    print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))
    print("")
    # Confusion Matrix
    print("Confusion Matrix")
    print("{0}".format(metrics.confusion_matrix(y_test, rf_predict_test)))
    print("")
    # Classification Report
    print("Classification Report")
    print(metrics.classification_report(y_test, rf_predict_test))

def parameter_tuning(x_train, x_test, y_train, y_test):
    """Setting regularization parameter"""
    C_start = 0.1
    C_end = 5
    C_inc = 0.1
    C_values, recall_scores =[], []
    C_val = C_start
    best_recall_score = 0
    while(C_val < C_end):
        C_values.append(C_val)
        lr_model_loop = LogisticRegression(C=C_val, class_weight="balanced", random_state=42)
        lr_model_loop.fit(x_train, y_train.ravel())
        lr_predict_loop_test=lr_model_loop.predict(x_test)
        recall_score=metrics.recall_score(y_test, lr_predict_loop_test)
        recall_scores.append(recall_score)
        if(recall_score > best_recall_score):
            best_recall_score = recall_score
        #   best_lr_predict_test = lr_predict_loop_test
            
        C_val = C_val + C_inc
        
    BEST_SCORE_C_VAL = C_values[recall_scores.index(best_recall_score)]
    print("first max value of {0:.3f} occurred at C={1:.3f}".format(best_recall_score, BEST_SCORE_C_VAL))
    # plot
    plt.plot(C_values, recall_scores)
    plt.title("Tuning C value")
    plt.xlabel("C value")
    plt.ylabel("recall score")
    return BEST_SCORE_C_VAL

def train_with_logisticregression(x_train, x_test, y_train, y_test, c, class_weight, random_state):
    """function for training with LogisticRegression algorithm"""
    lr_model = LogisticRegression(C=c, class_weight = class_weight, random_state = random_state)
    lr_model.fit(x_train, y_train.ravel())

    # Performance on Testing data
    # Predict values using the training data
    lr_predict_test = lr_model.predict(x_test)
    # Metrics
    print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
    print("")
    print("Confusion Matrix")
    print("{0}".format(metrics.confusion_matrix(y_test, lr_predict_test)))
    print("")
    print("Classification Report")
    print(metrics.classification_report(y_test, lr_predict_test))
    print(metrics.recall_score(y_test,lr_predict_test))
    return lr_model

def save_model(model_name):
    """Save model file"""
    joblib.dump(model_name, MODEL_PATH)
    print("Model file saved to: ", MODEL_PATH)

def train():
    df = load_data(DATA_PATH)
    clean_df = cleanup_data(df)
    x_train, x_test, y_train, y_test = split_data(clean_df)
    x_train, x_test = post_split_data_cleanup(x_train, x_test)
    lr_model = train_with_logisticregression(x_train, x_test, y_train, y_test, BEST_SCORE_C_VAL, "balanced", RANDOM_STATE)
    save_model(lr_model)

train()