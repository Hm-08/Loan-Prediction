import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    data.replace({'NA': np.nan, '': np.nan}, inplace=True)
    boolean_column = ['originated', 'approved']
    data[boolean_column] = data[boolean_column].replace({True: 1, False: 0})

    X = data.drop(columns=['loanId', 'anon_ssn', 'applicationDate', 'originatedDate', 'loanStatus', 'clarityFraudId', 'hasCF'], axis=1)
    y = data['loanStatus']

    X_num = X.drop(["payFrequency", "state", "leadType", "fpStatus"], axis=1)
    X_cat = X[["payFrequency", "state", "leadType", "fpStatus"]]

    # Handling missing values for numerical columns
    numeric_columns = ['apr', 'nPaidOff', 'loanAmount']
    X_num[numeric_columns] = X_num[numeric_columns].fillna(0)
    
    # Handling missing values for categorical columns
    X_cat = X_cat.copy()
    X_cat.loc[:, 'payFrequency'] = X_cat['payFrequency'].fillna('N') # N - Not applicable
    X_cat.loc[:, 'state'] = X_cat['state'].fillna('N') # N - Not applicable
    X_cat.loc[:, 'fpStatus'] = X_cat['fpStatus'].fillna('None')

    # Standardization
    scaler = StandardScaler(copy = False)
    scaler.fit(X_num)
    X_num_tr = scaler.transform(X_num)

    # One-hot encoding
    lb_encoder = LabelBinarizer()
    lb_encoder.fit(X_cat["payFrequency"])
    X_cat_payFrequency_tr = lb_encoder.transform(X_cat["payFrequency"]) # return as numpy

    lb_encoder.fit(X_cat["state"])
    X_cat_state_tr = lb_encoder.transform(X_cat["state"]) # return as numpy

    lb_encoder.fit(X_cat["leadType"])
    X_cat_leadType_tr = lb_encoder.transform(X_cat["leadType"]) # return as numpy

    lb_encoder.fit(X_cat["fpStatus"])
    X_cat_fpStatus_tr = lb_encoder.transform(X_cat["fpStatus"]) # return as numpy

    X_cat_tr = np.hstack([X_cat_payFrequency_tr, X_cat_state_tr, X_cat_leadType_tr, X_cat_fpStatus_tr])
    y.fillna('Not Available', inplace=True)
    le = LabelEncoder()
    le.fit(y)
    target = le.transform(y) # return as numpy
    features = np.hstack([X_num_tr, X_cat_tr])

    return features, target
