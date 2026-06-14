from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class LoanInput(BaseModel):
    application_id: Optional[str] = None

    no_of_dependents: int = Field(..., ge=0)

    education: Literal["Graduate", "Not Graduate"]

    self_employed: Literal["Yes", "No"]

    income_annum: int = Field(..., gt=0)

    loan_amount: int = Field(..., gt=0)

    loan_term: int = Field(..., gt=0)

    cibil_score: int = Field(..., ge=300, le=900)

    residential_assets_value: int

    commercial_assets_value: int

    luxury_assets_value: int

    bank_asset_value: int


class LoanBatchInput(BaseModel):
    data: List[LoanInput]


class PredictionResponse(BaseModel):
    application_id: str
    timestamp: str
    loan_status: str


class BatchItemResponse(BaseModel):
    application_id: Optional[str] = None
    timestamp: str
    loan_status: Optional[str] = None
    error: Optional[list] = None


class BatchResponse(BaseModel):
    results: List[BatchItemResponse]