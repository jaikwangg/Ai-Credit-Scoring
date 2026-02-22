import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "Credit Scoring API Gateway"}

def test_score_request_success():
    payload = {
        "request_id": "req-12345",
        "customer_id": "cust-existing-001",
        "demographics": {
            "age": 30,
            "employment_status": "Employed",
            "education_level": "Bachelor",
            "marital_status": "Single"
        },
        "financials": {
            "monthly_income": 5000.0,
            "monthly_expenses": 2000.0,
            "existing_debt": 1000.0
        },
        "loan_details": {
            "loan_amount": 20000.0,
            "loan_term_months": 36,
            "loan_purpose": "Auto"
        }
    }
    
    response = client.post("/api/v1/score/request", json=payload)
    
    # We should get a successful response
    assert response.status_code == 200, response.text
    data = response.json()
    
    assert data["request_id"] == "req-12345"
    assert "approved" in data
    assert "probability_score" in data
    assert "explanations" in data
    assert data["explanations"]["is_thin_file"] is False  # mocked logic says False for 'existing'

def test_score_request_thin_file():
    payload = {
        "request_id": "req-99999",
        "customer_id": "cust-new-999",
        "demographics": {
            "age": 22,
            "employment_status": "Student",
            "education_level": "High School",
            "marital_status": "Single"
        },
        "financials": {
            "monthly_income": 1000.0,
            "monthly_expenses": 500.0,
            "existing_debt": 0.0
        },
        "loan_details": {
            "loan_amount": 5000.0,
            "loan_term_months": 12,
            "loan_purpose": "Personal"
        }
    }
    
    response = client.post("/api/v1/score/request", json=payload)
    
    # We should get a successful response
    assert response.status_code == 200, response.text
    data = response.json()
    
    assert data["request_id"] == "req-99999"
    assert "explanations" in data
    assert data["explanations"]["is_thin_file"] is True  # mocked logic says True for not 'existing'

def test_score_request_validation_error():
    # Missing required 'customer_id', 'monthly_income', etc.
    payload = {
        "request_id": "req-bad",
        "demographics": {
            "age": 15, # Invalid age < 18
            "employment_status": "Employed"
        }
    }
    
    response = client.post("/api/v1/score/request", json=payload)
    
    # Custom validation handler returns 422
    assert response.status_code == 422
    data = response.json()
    assert "Validation Failed. See logs or DLQ." in data["detail"]
    assert "errors" in data
