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
    def check_expenses_vs_income(cls, v, info):
        import logging as _logging
        monthly_income = (info.data or {}).get("monthly_income")
        if monthly_income is not None and v > monthly_income:
            _logging.getLogger(__name__).warning(
                "monthly_expenses (%s) exceeds monthly_income (%s) — potential data entry error",
                v,
                monthly_income,
            )
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


class PlannerResult(BaseModel):
    mode: str = Field("", description="'approved_guidance' or 'improvement_plan'")
    decision: Dict[str, Any] = Field(default_factory=dict, description="Normalized decision object from planner.")
    result_th: str = Field("", description="Planner-rendered Thai text.")
    plan: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured improvement plan (present on improvement_plan mode).",
    )
    issup_score: Optional[int] = Field(None, description="[IsSup] groundedness score (when enabled).")
    issup_passed: Optional[bool] = Field(None, description="True when IsSup score passes threshold.")


class RAGResult(BaseModel):
    source_count: int = Field(0, description="Number of RAG evidence sources attached to the planner output.")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="RAG evidence objects returned to frontend.")


class ScoringResponse(BaseModel):
    request_id: str = Field(..., description="Echoes the request trace ID.")
    approved: bool = Field(..., description="Binary classification result (approve/reject).")
    probability_score: float = Field(..., description="Continuous probability score [0.0 - 1.0].")
    model_type: str = Field(
        "rule_based",
        description="Scoring engine type: 'rule_based' = calibrated weighted formula (research). 'ml' = trained ML model.",
    )
    explanations: ModelExplanations
    advice: Optional[PlannerAdvice] = Field(None, description="Thai-language advice from planner+RAG.")
    planner: Optional[PlannerResult] = Field(None, description="Structured planner payload for frontend rendering.")
    rag: Optional[RAGResult] = Field(None, description="Structured RAG payload for frontend rendering.")


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
    overdue: float = Field(0.0, ge=0.0, description="Number of overdue days (not THB amount). Used in scoring as overdue/90.")
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
    issup_score: Optional[int] = Field(None, description="[IsSup] groundedness score (1-5). Present only when use_issup=true.")
    issup_passed: Optional[bool] = Field(None, description="True if IsSup score >= 2 (plan is grounded). Present only when use_issup=true.")


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


class AdvisorProfile(BaseModel):
    """User profile for profile-conditioned advisory.

    All fields optional — the advisor uses whichever fields are present
    and ignores missing ones (thin-file applicants typically lack many).
    """
    salary_per_month: Optional[float] = Field(None, description="Monthly income in THB.")
    occupation: Optional[str] = Field(None, description="Employment type/job title.")
    employment_tenure_months: Optional[int] = Field(None, description="Months at current job.")
    marriage_status: Optional[str] = Field(None, description="Single / Married / Divorced / Widowed.")
    has_coapplicant: Optional[bool] = Field(None, description="Whether there is a co-borrower.")
    coapplicant_income: Optional[float] = Field(None, description="Co-borrower monthly income.")
    credit_score: Optional[int] = Field(None, description="Bureau credit score (300-850 range).")
    credit_grade: Optional[str] = Field(None, description="Credit grade letter (AA/BB/CC/DD/FF).")
    outstanding_debt: Optional[float] = Field(None, description="Total outstanding debt in THB.")
    overdue_days_max: Optional[int] = Field(
        None,
        description="Maximum days past due ever recorded (credit-bureau bucket: 0/15/30/60/90/120).",
    )
    loan_amount_requested: Optional[float] = Field(None, description="Requested loan amount in THB.")
    loan_term_years: Optional[float] = Field(None, description="Requested loan term in years.")
    interest_rate: Optional[float] = Field(None, description="Quoted interest rate %.")


class AdvisorRequest(BaseModel):
    """Request schema for POST /rag/advisor — profile-conditioned reasoning.

    Unlike /rag/query which only paraphrases retrieved chunks, this endpoint
    asks the LLM to:
      1. Extract eligibility requirements from the retrieved policy chunks
      2. Compare each requirement against the supplied user profile
      3. Return structured pass/fail per requirement + overall verdict + advice
    """
    question: str = Field(..., description="The user's natural-language question.")
    profile: AdvisorProfile = Field(..., description="Applicant profile to evaluate against the policy.")
    top_k: Optional[int] = Field(6, description="Number of policy chunks to retrieve.")
    use_multihop: Optional[bool] = Field(
        False,
        description=(
            "If true, decompose the question into sub-questions and retrieve "
            "context per sub-question (Approach 2). Improves recall on wide questions."
        ),
    )
    use_self_rag: Optional[bool] = Field(
        False,
        description=(
            "If true, run a Self-RAG-style reflection loop after the first answer "
            "to verify groundedness and optionally retry retrieval (Approach 3)."
        ),
    )


class AdvisorRequirementCheck(BaseModel):
    """One eligibility requirement evaluated against a profile."""
    requirement: str = Field(..., description="The requirement extracted from policy (Thai).")
    user_value: str = Field(..., description="The user's value for this requirement (or 'ไม่ระบุ').")
    status: str = Field(..., description="pass | fail | unknown | not_applicable")
    explanation: str = Field(..., description="Why this status — references policy text.")


class AdvisorReasoningTrace(BaseModel):
    """Diagnostic trace of which reasoning approaches were used."""
    used_multihop: bool = False
    sub_questions: List[str] = Field(default_factory=list)
    sources_per_hop: List[int] = Field(default_factory=list)
    total_sources_after_dedup: int = 0
    used_self_rag: bool = False
    issup_score: Optional[int] = Field(None, description="[IsSup] reflection score 1-5.")
    issup_passed: Optional[bool] = None
    self_rag_retried: bool = False
    elapsed_seconds: float = 0.0


class AdvisorResponse(BaseModel):
    """Response from POST /rag/advisor"""
    question: str
    verdict: str = Field(..., description="overall: eligible | partially_eligible | ineligible | needs_more_info")
    verdict_summary: str = Field(..., description="Short Thai summary of the verdict and reason.")
    requirement_checks: List[AdvisorRequirementCheck] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list, description="Next steps the user can take to improve eligibility.")
    sources: List[RAGSource] = Field(default_factory=list)
    raw_answer: Optional[str] = Field(None, description="Raw LLM output for debugging.")
    reasoning_trace: Optional[AdvisorReasoningTrace] = Field(
        None,
        description="Diagnostic trace showing which reasoning approaches ran (multi-hop, Self-RAG, etc).",
    )


class SelfRAGTraceSchema(BaseModel):
    """Diagnostic trace from Self-RAG reflection steps."""
    retrieve_needed: bool = True
    nodes_before_isrel: int = 0
    nodes_after_isrel: int = 0
    isrel_scores: List[Dict[str, Any]] = Field(default_factory=list)
    issup_score: Optional[int] = None
    issup_passed: bool = False
    isgen_score: Optional[int] = Field(None, description="[IsGen] score (1-5): does the answer address the question?")
    isgen_passed: bool = Field(False, description="True if IsGen score >= threshold (answer is on-topic).")
    retry_attempted: bool = False
    resynth_used: bool = False
    total_reflection_calls: int = 0
    elapsed_s: float = 0.0


class RAGQueryResponse(BaseModel):
    """Response from POST /rag/query"""
    question: str
    answer: str
    router_label: str = Field(..., description="Detected query domain/category.")
    sources: List[RAGSource] = Field(default_factory=list, description="Retrieved source documents.")
    retrieved_count: int = Field(0, description="Total nodes retrieved before filtering.")
    validated_count: int = Field(0, description="Nodes passed validation and used in answer.")
    self_rag_trace: Optional[SelfRAGTraceSchema] = Field(None, description="Self-RAG reflection trace (only present on /rag/query/self).")


class SimplePlanRequest(BaseModel):
    """Request schema for POST /plan/simple.
    Accepts flat user features — model score and SHAP are computed internally.
    """
    request_id: str = Field(..., description="Unique trace ID.")
    features: UserInputFeatures


# -----------------------------------
# What-If / Counterfactual Simulation
# -----------------------------------

class WhatIfChange(BaseModel):
    """Describes a single feature change.  Provide exactly one of: value / delta / delta_pct."""
    value: Optional[Any] = Field(None, description="Set feature to this exact value.")
    delta: Optional[float] = Field(None, description="Add this amount to the current value (numeric features only).")
    delta_pct: Optional[float] = Field(None, description="Change by this percentage (numeric features only). E.g. -20 means -20%.")


class SimulationRequest(BaseModel):
    """Request schema for POST /plan/simulate."""
    request_id: str = Field(..., description="Unique trace ID.")
    features: UserInputFeatures
    what_if: Dict[str, WhatIfChange] = Field(
        ...,
        description="Feature changes to simulate. Keys must be valid UserInputFeatures field names.",
    )


class ScenarioResult(BaseModel):
    """Scoring outcome for one scenario (baseline or simulated)."""
    approved: bool
    p_approve: float
    p_reject: float
    shap_values: Dict[str, float]
    features: Dict[str, Any] = Field(default_factory=dict, description="Actual feature values used in this scenario.")


class SimulationResponse(BaseModel):
    """Response from POST /plan/simulate."""
    request_id: str
    baseline: ScenarioResult
    simulated: ScenarioResult
    delta_p_approve: float = Field(..., description="Change in approval probability (simulated − baseline).")
    shap_diff: Dict[str, float] = Field(default_factory=dict, description="Per-feature SHAP change (simulated − baseline).")
    changed_features: List[str] = Field(default_factory=list, description="Features that were modified by what_if.")
    verdict: str = Field(..., description="Thai-language summary of the simulation impact.")


# -----------------------------------
# Batch Evaluation
# -----------------------------------

class BatchItem(BaseModel):
    """A single applicant entry inside a batch request."""
    request_id: str = Field(..., description="Unique trace ID for this item.")
    features: UserInputFeatures


class BatchPlanRequest(BaseModel):
    """Request schema for POST /plan/batch."""
    batch_id: str = Field(..., min_length=1, description="Unique identifier for this batch job.")
    items: List[BatchItem] = Field(..., min_length=1, max_length=200, description="List of applicants to evaluate (max 200).")
    include_plan: bool = Field(False, description="If true, generate full Thai-language plan for each item (slower).")


class BatchItemResult(BaseModel):
    """Result for a single applicant in a batch."""
    request_id: str
    approved: bool
    p_approve: float
    p_reject: float
    mode: Optional[str] = Field(None, description="'approved_guidance' or 'improvement_plan'. Present only when include_plan=true.")
    result_th: Optional[str] = Field(None, description="Thai-language plan. Present only when include_plan=true.")
    error: Optional[str] = Field(None, description="Error message if this item failed.")


class BatchSummary(BaseModel):
    """Aggregate statistics for the batch."""
    total: int
    approved_count: int
    rejected_count: int
    error_count: int
    approval_rate: float = Field(..., description="Fraction of non-errored items that were approved.")
    avg_p_approve: float = Field(..., description="Mean approval probability across non-errored items.")
    elapsed_s: float


class BatchPlanResponse(BaseModel):
    """Response from POST /plan/batch."""
    batch_id: str
    summary: BatchSummary
    results: List[BatchItemResult]
