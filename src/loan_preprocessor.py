from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class LoanPreprocessor(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # clean column names
        df.columns = df.columns.str.strip()

        # drop id
        if 'loan_id' in df.columns:
            df = df.drop(columns=['loan_id'])

        # fix negative asset values
        asset_columns = [
            'residential_assets_value',
            'commercial_assets_value',
            'luxury_assets_value',
            'bank_asset_value'
        ]

        for col in asset_columns:
            median_val = df.loc[df[col] >= 0, col].median()
            df[col] = np.where(df[col] < 0, median_val, df[col])

        # encoding
        df['education'] = df['education'].str.strip().map({
            'Graduate': 1,
            'Not Graduate': 0
        })

        df['self_employed'] = df['self_employed'].str.strip().map({
            'Yes': 1,
            'No': 0
        })

        # feature engineering
        df['total_assets'] = (
            df['residential_assets_value']
            + df['commercial_assets_value']
            + df['luxury_assets_value']
            + df['bank_asset_value']
        )

        df['loan_to_income_ratio'] = (
            df['loan_amount'] / (df['income_annum'] + 1)
        )

        df['annual_payment'] = (
            df['loan_amount'] / df['loan_term']
        )

        df['debt_service_ratio'] = (
            df['annual_payment'] / (df['income_annum'] + 1)
        )

        df['assets_to_loan_ratio'] = (
            df['total_assets'] / (df['loan_amount'] + 1)
        )

        df['income_per_dependent'] = (
            df['income_annum'] / (df['no_of_dependents'] + 1)
        )

        return df