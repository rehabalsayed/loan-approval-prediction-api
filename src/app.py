from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import logging
import uuid
from datetime import datetime
from typing import List, Dict
from pydantic import ValidationError
from pathlib import Path

from src.database import conn, cursor
from src.loan_preprocessor import LoanPreprocessor
from src.schemas import LoanBatchInput, LoanInput, BatchResponse, BatchItemResponse

from fastapi import Body

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Loan Approval API")



BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "final_loan_model.pkl"

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Loan Approval API Running"}


# =========================
# SINGLE PREDICTION
# =========================
@app.post("/predict")
def predict(data: LoanInput):

    try:
        input_data = data.model_dump()

        application_id = (
            input_data.get("application_id")
            or str(uuid.uuid4())
        )

        input_data.pop(
            "application_id",
            None
        )

        df = pd.DataFrame([input_data])

        prediction = model.predict(df)[0]

        timestamp = datetime.now().isoformat()

        # save prediction to database
        cursor.execute(
            """
            INSERT INTO predictions
            (
                application_id,
                timestamp,
                loan_status
            )
            VALUES (?, ?, ?)
            """,
            (
                application_id,
                timestamp,
                str(prediction).strip()
            )
        )

        conn.commit()

        return {
            "application_id": application_id,
            "timestamp": timestamp,
            "loan_status": str(prediction).strip()
        }

    except Exception as e:

        logging.error(str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================
# BATCH PREDICTION (FIXED + PARTIAL SUCCESS)
# =========================
@app.post(
    "/predict_batch",
    response_model=BatchResponse
)
def predict_batch(
    payload: dict = Body(
        ...,
        example={
            "data": [
                {
                    "application_id": "APP_1",
                    "no_of_dependents": 2,
                    "education": "Graduate",
                    "self_employed": "No",
                    "income_annum": 500000,
                    "loan_amount": 200000,
                    "loan_term": 36,
                    "cibil_score": 750,
                    "residential_assets_value": 150000,
                    "commercial_assets_value": 50000,
                    "luxury_assets_value": 20000,
                    "bank_asset_value": 100000
                }
            ]
        }
    )
):


    results = []

    for item in payload["data"]:

        application_id = item.get("application_id") or str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        try:
            
            validated = LoanInput(**item)
        
            input_data = validated.model_dump()
            input_data.pop("application_id", None)

            df = pd.DataFrame([input_data])

            prediction = model.predict(df)[0]

            # SAVE TO DB
            cursor.execute(
                """
                INSERT INTO predictions
                (application_id, timestamp, loan_status)
                VALUES (?, ?, ?)
                """,
                (
                    application_id,
                    timestamp,
                    str(prediction).strip()
                )
            )

            results.append(
                BatchItemResponse(
                    application_id=application_id,
                    timestamp=timestamp,
                    loan_status=str(prediction).strip(),
                    error=None
                )
            )

            logging.info(
                f"SUCCESS: {application_id} -> {prediction}"
            )

        except Exception as e:
            results.append(
                BatchItemResponse(
                    application_id=application_id,
                    timestamp=timestamp,
                    loan_status=None,
                    error=[str(e)]
                )
            )

    conn.commit()

    return BatchResponse(results=results)