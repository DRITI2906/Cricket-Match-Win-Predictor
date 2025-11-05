from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class MatchInput(BaseModel):
    team1: str = Field(..., description="First team name (batting team)")
    team2: str = Field(..., description="Second team name (bowling team)")
    venue: str = Field(..., description="Match venue")
    toss_winner: Optional[str] = Field(None, description="Toss winner team")
    toss_decision: Optional[str] = Field(None, description="Bat or Bowl")
    match_type: str = Field(default="ODI", description="Match type (ODI, T20, Test)")
    
    # Optional match context fields for more accurate predictions
    runs_required: Optional[int] = Field(None, description="Runs required to win")
    balls_remaining: Optional[int] = Field(None, description="Balls remaining in the innings")
    wickets_in_hand: Optional[int] = Field(None, description="Wickets in hand for batting team")
    target_match: Optional[int] = Field(None, description="Target score for the match")
    current_run_rate: Optional[float] = Field(None, description="Current run rate")
    required_run_rate: Optional[float] = Field(None, description="Required run rate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "team1": "India",
                "team2": "Australia",
                "venue": "Melbourne Cricket Ground",
                "toss_winner": "India",
                "toss_decision": "bat",
                "match_type": "ODI"
            }
        }

class ShapValue(BaseModel):
    feature: str
    value: float
    impact: str  # positive, negative, neutral

class PredictionResponse(BaseModel):
    winner: str
    probability: float
    confidence: str  # high, medium, low
    shap_explanation: List[ShapValue]
    factors: Dict[str, str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "winner": "India",
                "probability": 0.65,
                "confidence": "high",
                "shap_explanation": [
                    {"feature": "Home Advantage", "value": 0.15, "impact": "positive"},
                    {"feature": "Recent Form", "value": 0.12, "impact": "positive"}
                ],
                "factors": {
                    "toss": "Won by India",
                    "venue": "Favorable conditions"
                }
            }
        }
