
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.services.model_runner import ModelRunnerService
from src.api.schemas.payload import ScoringRequest, DemographicData, FinancialData, LoanRequestData
from src.planner.planning import generate_response
from src.planner.rag_bridge import build_user_input, build_shap_json

def test_planner_with_model_output():
    # High risk case: bad credit score (400)
    payload = ScoringRequest(
        request_id="test-1",
        customer_id="cust-1",
        demographics=DemographicData(age=30, employment_status="Employed", education_level="Bachelor", marital_status="Single"),
        financials=FinancialData(monthly_income=50000, monthly_expenses=20000, existing_debt=0),
        loan_details=LoanRequestData(loan_amount=1000000, loan_term_months=360, loan_purpose="Mortgage")
    )
    
    merged_features = {
        "credit_bureau_score": 400, # BAD
        "credit_grade": "FF",       # BAD
        "outstanding": 0.0,
        "overdue_amount": 0.0,
        "has_coapplicant": False,
        "is_thin_file": False
    }
    
    model_result = ModelRunnerService.run_inference(merged_features, payload)
    print(f"Model Approved: {model_result['approved']}")
    print(f"Model SHAP: {model_result['shap_values']}")
    
    user_input = build_user_input(payload, merged_features)
    risk_prob = model_result["probability_score"]
    model_output = {
        "prediction": 1 if model_result["approved"] else 0,
        "probabilities": {
            "1": round(1.0 - risk_prob, 4),  # P(approved)
            "0": round(risk_prob, 4),          # P(default)
        },
    }
    shap_json = build_shap_json(model_result["shap_values"])
    
    # Run planner without RAG for simplicity
    plan_result = generate_response(user_input, model_output, shap_json, rag_lookup=None)
    
    print(f"Planner Mode: {plan_result['mode']}")
    print(f"Planner Top Negatives: {plan_result['plan']['risk_drivers']['top_negative']}")
    print(f"Planner Result Text:\n{plan_result['result_th']}")

if __name__ == "__main__":
    test_planner_with_model_output()
