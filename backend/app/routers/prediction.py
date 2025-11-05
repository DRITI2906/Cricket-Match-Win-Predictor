from fastapi import APIRouter, HTTPException
import logging
from app.models.match import MatchInput, PredictionResponse
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)
router = APIRouter()
# Lazy-initialize the service to avoid import-time failures during deployment
prediction_service = None

@router.post("/predict", response_model=PredictionResponse)
async def predict_match(match_data: MatchInput):
    """
    Predict the outcome of a cricket match
    """
    try:
        global prediction_service
        if prediction_service is None:
            try:
                prediction_service = PredictionService()
            except Exception as e:
                logger.exception("Failed to initialize PredictionService")
                raise HTTPException(status_code=500, detail="Prediction service unavailable")

        result = await prediction_service.predict(match_data)
        return result
    except Exception as e:
        # Log the full exception with stack trace so deployments show useful logs
        logger.exception("Unhandled error in /api/predict")
        # Return a generic HTTP 500 with minimal detail
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health():
    """Simple health endpoint reporting model readiness"""
    global prediction_service
    if prediction_service is None:
        try:
            prediction_service = PredictionService()
        except Exception:
            logger.exception("Failed to initialize PredictionService during health check")
            return {"ready": False, "model_loaded": False}

    model_loaded = getattr(prediction_service, 'model_loaded', False)
    return {"ready": True, "model_loaded": bool(model_loaded)}