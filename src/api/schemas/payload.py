from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, conint, condecimal, field_validator

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
    
    @field_validator("monthly_expenses")
    @classmethod
    def check_expenses_vs_income(cls, v):
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

class PlannerAdvice(BaseModel):
    mode: str = Field("", description="'approved_guidance' or 'improvement_plan'")
    result_th: str = Field("", description="Thai-language plan or approval checklist.")
    rag_sources: List[Dict[str, Any]] = Field(default_factory=list, description="RAG evidence used in advice.")


class ScoringResponse(BaseModel):
    request_id: str = Field(..., description="Echoes the request trace ID.")
    approved: bool = Field(..., description="Binary classification result (approve/reject).")
    probability_score: float = Field(..., description="Continuous probability score [0.0 - 1.0].")
    explanations: ModelExplanations
    advice: Optional[PlannerAdvice] = Field(None, description="Thai-language advice from planner+RAG.")


# -----------------------------------
# External Plan Request Schemas
# (for bring-your-own-model integration)
# -----------------------------------

class UserInputFeatures(BaseModel):
    """Flat feature dict matching planner's DRIVER_QUERY_MAP keys.
    Accepts the same format as the test cases (Salary, credit_score, etc.)
    """
    Salary: float = Field(..., description="Monthly income (THB).")
    Occupation: Optional[str] = Field("Unknown", description="E.g., Salaried_Employee, Freelancer.")
    Marriage_Status: Optional[str] = Field("Unknown", description="E.g., Single, Married.")
    credit_score: float = Field(..., description="Credit bureau score (300–850).")
    credit_grade: str = Field("CC", description="Credit grade: AA, BB, CC, DD, EE, FF.")
    outstanding: float = Field(0.0, description="Total outstanding debt (THB).")
    overdue: float = Field(0.0, description="Overdue amount in days or THB.")
    Coapplicant: Union[bool, int] = Field(False, description="1/true if co-applicant exists.")
    loan_amount: float = Field(..., description="Requested loan amount (THB).")
    loan_term: float = Field(..., description="Loan term in years.")
    Interest_rate: Optional[float] = Field(None, description="Interest rate (%). Optional.")


class ModelOutputPayload(BaseModel):
    """Prediction result from an external ML model."""
    prediction: int = Field(..., description="0 = rejected, 1 = approved.")
    probabilities: Dict[str, float] = Field(
        ...,
        description='Probability per class. Keys "0" and "1". e.g. {"0": 0.68, "1": 0.32}',
    )


class ShapPayload(BaseModel):
    """SHAP values from an external ML model (approval-probability convention).
    Sign convention: negative = feature HURTS approval, positive = HELPS approval.
    Feature names must match planner's DRIVER_QUERY_MAP keys:
    credit_score, credit_grade, outstanding, overdue, loan_amount, loan_term, Salary, Interest_rate
    """
    base_value: float = Field(0.5, description="SHAP base value (expected model output).")
    values: Dict[str, float] = Field(..., description="Feature name → SHAP contribution.")


class ExternalPlanRequest(BaseModel):
    """Request schema for /plan/external endpoint.
    Accepts user features + model output + SHAP values from an external ML model,
    bypassing the internal FeatureMerger and ModelRunner.
    """
    request_id: str = Field(..., description="Unique trace ID.")
    user_input: UserInputFeatures
    model_output: ModelOutputPayload
    shap_json: ShapPayload


class ExternalPlanResponse(BaseModel):
    """Response from /plan/external endpoint."""
    request_id: str
    mode: str = Field(..., description="'approved_guidance' or 'improvement_plan'.")
    approved: bool
    p_approve: float
    p_reject: float
    result_th: str = Field(..., description="Thai-language plan or approval checklist.")
    rag_sources: List[Dict[str, Any]] = Field(default_factory=list)


# -----------------------------------
# RAG Direct Query Schemas
# -----------------------------------

class RAGSource(BaseModel):
    title: str = Field("Unknown", description="Document title.")
    category: str = Field("Uncategorized", description="Document category.")
    institution: Optional[str] = Field(None, description="Issuing institution.")
    score: Optional[float] = Field(None, description="Similarity score.")


class RAGQueryRequest(BaseModel):
    """Request schema for POST /rag/query"""
    question: str = Field(..., description="Question to ask the RAG system.")
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve (default: settings.SIMILARITY_TOP_K).")


class RAGQueryResponse(BaseModel):
    """Response from POST /rag/query"""
    question: str
    answer: str
    router_label: str = Field(..., description="Detected query domain/category.")
    sources: List[RAGSource] = Field(default_factory=list, description="Retrieved source documents.")
    retrieved_count: int = Field(0, description="Total nodes retrieved before filtering.")
    validated_count: int = Field(0, description="Nodes passed validation and used in answer.")


class SimplePlanRequest(BaseModel):
    """Request schema for POST /plan/simple.
    Accepts flat user features — model score and SHAP are computed internally.
    """
    request_id: str = Field(..., description="Unique trace ID.")
    features: UserInputFeatures
