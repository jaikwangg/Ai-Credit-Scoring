from typing import Optional, Dict
from pydantic import BaseModel, Field, conint, condecimal, validator

# -----------------------------------
# Incoming Request Schemas
# -----------------------------------

class DemographicData(BaseModel):
    age: conint(ge=18, le=120) = Field(..., description="Age of the applicant in years.")
    employment_status: str = Field(..., description="E.g., Employed, Self-Employed, Unemployed.")
    education_level: Optional[str] = Field("Unknown", description="E.g., Bachelor, Master, High School.")
    marital_status: Optional[str] = Field("Unknown", description="E.g., Single, Married.")

class FinancialData(BaseModel):
    monthly_income: condecimal(ge=0, decimal_places=2) = Field(..., description="Monthly income in local currency.")
    monthly_expenses: condecimal(ge=0, decimal_places=2) = Field(..., description="Monthly expenses in local currency.")
    existing_debt: condecimal(ge=0, decimal_places=2) = Field(0.00, description="Total existing debt.")
    
    @validator("monthly_expenses")
    def check_expenses_vs_income(cls, v, values):
        # We allow expenses > income but we could log/warn
        return v

class LoanRequestData(BaseModel):
    loan_amount: condecimal(gt=0, decimal_places=2) = Field(..., description="Requested loan amount.")
    loan_term_months: conint(gt=0, le=360) = Field(..., description="Duration of the loan in months.")
    loan_purpose: str = Field(..., description="E.g., Mortgage, Personal, Auto.")

class ScoringRequest(BaseModel):
    request_id: str = Field(..., description="Unique Trace ID from the API Gateway/Client.")
    customer_id: str = Field(..., description="Internal Customer Identifier. Allows linking to historical data.")
    demographics: DemographicData
    financials: FinancialData
    loan_details: LoanRequestData

# -----------------------------------
# Outgoing Response Schemas
# -----------------------------------

class ModelExplanations(BaseModel):
    is_thin_file: bool = Field(False, description="True if no historical DB data was found.")
    tree_shap_values: Dict[str, float] = Field(default_factory=dict, description="SHAP feature attributions.")

class ScoringResponse(BaseModel):
    request_id: str = Field(..., description="Echoes the request trace ID.")
    approved: bool = Field(..., description="Binary classification result (approve/reject).")
    probability_score: float = Field(..., description="Continuous probability score [0.0 - 1.0].")
    explanations: ModelExplanations
