from fastapi import APIRouter, HTTPException
from app.models.match import MatchInput, PredictionResponse
from app.services.prediction_service import PredictionService

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
        raise HTTPException(status_code=500, detail=str(e))
