print("SCRIPT IS RUNNING")
 
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


import pandas as pd
import numpy as np

import os
import joblib

from sklearn.preprocessing import StandardScaler

from  src.loan_preprocessor import LoanPreprocessor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_PATH = BASE_DIR / "data" / "loan_approval_dataset.csv"


df = pd.read_csv(RAW_DATA_PATH)

df.columns = df.columns.str.strip()
   
    
final_model = Pipeline(steps=[
    ('preprocess', LoanPreprocessor()),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=1,
        random_state=42
    ))
])
  
     
X = df.drop(columns=['loan_status'])
y = df['loan_status']

final_model.fit(X, y)


joblib.dump(final_model, 'final_loan_model.pkl')

print("Model saved successfully")

