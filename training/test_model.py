import joblib
import pandas as pd

from src.loan_preprocessor import LoanPreprocessor

model = joblib.load("final_loan_model.pkl")

sample = pd.DataFrame({
    'no_of_dependents': [2],
    'education': ['Graduate'],
    'self_employed': ['No'],
    'income_annum': [500000],
    'loan_amount': [200000],
    'loan_term': [12],
    'cibil_score': [750],
    'residential_assets_value': [1000000],
    'commercial_assets_value': [500000],
    'luxury_assets_value': [200000],
    'bank_asset_value': [300000]
})

prediction = model.predict(sample)

print(prediction)




