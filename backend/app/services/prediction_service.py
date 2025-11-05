from app.models.match import MatchInput, PredictionResponse, ShapValue
from app.ml.predictor import CricketPredictor
from typing import List

class PredictionService:
    """
    Service for cricket match prediction with ML model
    """
    
    def __init__(self):
        
        try:
            self.predictor = CricketPredictor()
            print("ML Predictor initialized successfully")
        except Exception as e:
            print(f"Error initializing predictor: {e}")
            self.predictor = None
    
    async def predict(self, match_data: MatchInput) -> PredictionResponse:
        """
        Predict match outcome based on input data using ML model
        """
       
        model_input = {
            'team1': match_data.team1,
            'team2': match_data.team2,
            'batting_team': match_data.team1,  # Assume team1 is batting
            'bowling_team': match_data.team2,
            'venue': match_data.venue,
            'toss_winner': match_data.toss_winner or match_data.team1,
            'toss_decision': match_data.toss_decision or 'bat',
            # These would come from match context in a real scenario
            'runs_required': getattr(match_data, 'runs_required', 150),
            'balls_remaining': getattr(match_data, 'balls_remaining', 120),
            'wickets_in_hand': getattr(match_data, 'wickets_in_hand', 10),
            'target_match': getattr(match_data, 'target_match', 250),
            'current_run_rate': getattr(match_data, 'current_run_rate', 6.0),
            'required_run_rate': getattr(match_data, 'required_run_rate', 7.5)
        }
        
       
        if self.predictor:
            winner, probability, shap_values = self.predictor.predict(model_input)
        else:
            # Fallback to mock prediction
            import random
            probability = random.uniform(0.55, 0.85)
            winner = match_data.team1
            shap_values = self._default_shap_values()
        
    
        confidence = "high" if probability > 0.7 else "medium" if probability > 0.6 else "low"
        
        # Convert SHAP values to response format
        shap_explanation = [
            ShapValue(
                feature=sv['feature'],
                value=sv['value'],
                impact=sv['impact']
            ) for sv in shap_values
        ]
        
        factors = {
            "toss": f"Won by {match_data.toss_winner}" if match_data.toss_winner else "N/A",
            "toss_decision": match_data.toss_decision if match_data.toss_decision else "N/A",
            "venue": match_data.venue,
            "match_type": match_data.match_type
        }
        
        return PredictionResponse(
            winner=winner,
            probability=round(probability, 2),
            confidence=confidence,
            shap_explanation=shap_explanation,
            factors=factors
        )
    
    def _default_shap_values(self) -> List[dict]:
        """
        Default SHAP values when model is not available
        """
        return [
            {'feature': 'Team Strength', 'value': 0.15, 'impact': 'positive'},
            {'feature': 'Home Advantage', 'value': 0.12, 'impact': 'positive'},
            {'feature': 'Recent Form', 'value': 0.10, 'impact': 'positive'},
            {'feature': 'Toss Impact', 'value': -0.05, 'impact': 'negative'},
            {'feature': 'Venue History', 'value': 0.08, 'impact': 'positive'},
        ]
