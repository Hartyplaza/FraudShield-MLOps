"""
src/fraudshield/serving/schema.py

Pydantic request and response schemas for the FraudShield V2 API.
These define the exact contract between the API and its callers.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    """
    Input schema for a single transaction to be scored.

    The 28 feature columns match FEATURE_COLS in src/fraudshield/features/engineer.py.
    Temporal and behavioural features should be pre-computed by the caller
    or by a feature service before hitting this endpoint.
    """

    # Temporal
    hour              : int   = Field(..., ge=0, le=23)
    day_of_week       : int   = Field(..., ge=0, le=6)
    is_night          : int   = Field(0, ge=0, le=1)
    is_weekend        : int   = Field(0, ge=0, le=1)
    is_business_hours : int   = Field(0, ge=0, le=1)

    # Amount
    amount            : float = Field(..., gt=0)
    amount_z_score    : float = 0.0
    is_high_amount    : int   = Field(0, ge=0, le=1)
    amount_vs_user_max: float = 0.0
    user_mean_amount  : float = 0.0
    user_std_amount   : float = 0.0
    user_max_amount   : float = 0.0

    # Velocity
    txn_count_1d  : float = 0.0
    txn_sum_1d    : float = 0.0
    txn_count_7d  : float = 0.0
    txn_sum_7d    : float = 0.0
    txn_count_30d : float = 0.0
    txn_sum_30d   : float = 0.0

    # Behavioural
    device_changed            : int   = Field(0, ge=0, le=1)
    location_changed          : int   = Field(0, ge=0, le=1)
    time_since_last_txn_hrs   : float = 0.0

    # Missing flags
    location_missing : int = Field(0, ge=0, le=1)
    device_missing   : int = Field(0, ge=0, le=1)

    # Encoded categoricals
    txn_type_encoded   : int = 0
    location_encoded   : int = 0
    device_encoded     : int = 0
    log_format_encoded : int = 0
    currency_encoded   : int = 0

    class Config:
        json_schema_extra = {
            "example": {
                "hour"              : 2,
                "day_of_week"       : 5,
                "is_night"          : 1,
                "is_weekend"        : 1,
                "is_business_hours" : 0,
                "amount"            : 4800.0,
                "amount_z_score"    : 3.2,
                "is_high_amount"    : 1,
                "amount_vs_user_max": 0.96,
                "user_mean_amount"  : 2400.0,
                "user_std_amount"   : 1200.0,
                "user_max_amount"   : 4997.0,
                "txn_count_1d"      : 1.0,
                "txn_sum_1d"        : 4800.0,
                "txn_count_7d"      : 5.0,
                "txn_sum_7d"        : 12000.0,
                "txn_count_30d"     : 18.0,
                "txn_sum_30d"       : 43000.0,
                "device_changed"    : 1,
                "location_changed"  : 0,
                "time_since_last_txn_hrs": 0.5,
                "location_missing"  : 1,
                "device_missing"    : 0,
                "txn_type_encoded"  : 0,
                "location_encoded"  : 3,
                "device_encoded"    : 1,
                "log_format_encoded": 2,
                "currency_encoded"  : 1,
            }
        }


class FraudResponse(BaseModel):
    """
    Output schema for a fraud scoring result.
    """
    fraud_probability : float
    is_fraud          : bool
    risk_level        : str
    threshold_used    : float
    top_features      : dict


class HealthResponse(BaseModel):
    """
    Response schema for the /health endpoint.
    """
    status       : str
    model_loaded : bool
    threshold    : float
    version      : str
