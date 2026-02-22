from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime
from src.db.database import Base
from datetime import datetime

class CreditScoreResult(Base):
    __tablename__ = "credit_score_results"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, unique=True, index=True)
    customer_id = Column(String, index=True)
    
    # Model Outputs
    approved = Column(Boolean)
    probability_score = Column(Float)
    is_thin_file = Column(Boolean)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, index=True)
    customer_id = Column(String, index=True)
    action = Column(String) # e.g., 'SCORE_REQUESTED', 'DECISION_MADE'
    details = Column(String) # JSON payload stringified
    timestamp = Column(DateTime, default=datetime.utcnow)
