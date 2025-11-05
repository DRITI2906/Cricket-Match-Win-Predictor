from fastapi import APIRouter, HTTPException
import logging
from app.models.match import MatchInput, PredictionResponse
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)
router = APIRouter()
prediction_service = PredictionService()

@router.post("/predict", response_model=PredictionResponse)
async def predict_match(match_data: MatchInput):
    """
    Predict the outcome of a cricket match
    """
    try:
        result = await prediction_service.predict(match_data)
        return result
    except Exception as e:
        
        logger.exception("Unhandled error in /api/predict")
       
        raise HTTPException(status_code=500, detail="Internal server error")