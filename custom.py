#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import os
import pandas as pd
import numpy as np
from typing import List, Optional
from catboost import CatBoostClassifier
import io

def read_input_data(input_binary_data):
    return pd.read_csv(io.BytesIO(input_binary_data))

def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    class_order: Optional[List[str]] = None,
    row_weights: Optional[np.ndarray] = None,
    **kwargs,
) -> None:

    #Take advantage of transform function below - This might not be applicable to your use case, depending on the preprocessing you do.
    X = transform(X,model=None)

    estimator = CatBoostClassifier(iterations=10,
                               depth=2,
                               learning_rate=1,
                               loss_function='Logloss',
                               verbose=True)

    cat_features = ['term', 'int_rate', 'grade', 'sub_grade', 'emp_length', 'emp_title','desc','purpose','title','home_ownership', 'verification_status', 'pymnt_plan', 'loan_id', 'zip_code', 'addr_state', 'initial_list_status']

    # train the model
    estimator.fit(X,y,cat_features)
    
    #Dumping the model in output_dir --> DataRobot will automatically find pkl files saved there.
    pickle.dump(estimator, open('{}/model.pkl'.format(output_dir), 'wb'))


def transform(data,model):
    data.drop(['earliest_cr_line','url'], axis=1, inplace=True)
    cat_features = ['term', 'int_rate', 'grade', 'sub_grade', 'emp_length', 'emp_title','desc','purpose','title','home_ownership', 'verification_status', 'pymnt_plan', 'loan_id', 'zip_code', 'addr_state', 'initial_list_status']
    # Fill null values for Categorical Features
    for c in cat_features:
        data[c] = data[c].fillna('unknown')
        try:
            data[c] = data[c].astype(str)
        except:
            pass
    # Fill null values for numerical Features
    data = data.fillna(0)
    return data

def score(data, model, **kwargs):

    results = model.predict_proba(data)

    #Create two columns with probability results
    predictions = pd.DataFrame({'1': results[:, 0]})
    predictions['0'] = 1 - predictions['1']

    return predictions

