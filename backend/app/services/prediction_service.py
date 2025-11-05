import logging
from app.models.match import MatchInput, PredictionResponse, ShapValue
from app.ml.predictor import CricketPredictor
from typing import List

logger = logging.getLogger(__name__)

class PredictionService:
    
    
    def _init_(self):
       
        self.predictor = None
        try:
            logger.info("Initializing CricketPredictor...")
            self.predictor = CricketPredictor()
            logger.info("PredictionService initialized with ML predictor")
        except Exception:
           
            logger.exception("Failed to initialize CricketPredictor during PredictionService startup")
       
        try:
            self.model_loaded = bool(getattr(self.predictor, 'model', None))
            logger.info(f"Model loaded: {self.model_loaded}")
            if self.model_loaded:
                
                model_info = getattr(self.predictor, 'model_info', None)
                if model_info:
                    logger.info(f"Model info keys: {list(model_info.keys())}")
        except Exception:
            self.model_loaded = False
            logger.exception("Error determining model_loaded flag")
    
    async def predict(self, match_data: MatchInput) -> PredictionResponse:
       
        model_input = {
            'team1': match_data.team1,
            'team2': match_data.team2,
            'batting_team': match_data.team1,  # Assume team1 is batting
            'bowling_team': match_data.team2,
            'venue': match_data.venue,
            'toss_winner': match_data.toss_winner or match_data.team1,
            'toss_decision': match_data.toss_decision or 'bat',
            
            'runs_required': getattr(match_data, 'runs_required', 150),
            'balls_remaining': getattr(match_data, 'balls_remaining', 120),
            'wickets_in_hand': getattr(match_data, 'wickets_in_hand', 10),
            'target_match': getattr(match_data, 'target_match', 250),
            'current_run_rate': getattr(match_data, 'current_run_rate', 6.0),
            'required_run_rate': getattr(match_data, 'required_run_rate', 7.5)
        }
        
        
        if getattr(self, "predictor", None):
            winner, batting_win_prob, shap_values = self.predictor.predict(model_input)
        else:
           
            logger.warning("Predictor not available, returning fallback prediction")
            batting_team = model_input.get('batting_team') or match_data.team1
            winner = batting_team
            batting_win_prob = 0.5
            shap_values = self._generate_dynamic_shap_values(model_input)

        
        confidence = "high" if batting_win_prob > 0.7 else "medium" if batting_win_prob > 0.6 else "low"
        
       
        shap_explanation = []
        try:
            for sv in shap_values:
            
                feature = str(sv.get('feature', 'Unknown'))
                value = float(sv.get('value', 0.0))
                impact = str(sv.get('impact', 'neutral'))
                shap_explanation.append(ShapValue(feature=feature, value=value, impact=impact))
        except Exception:
            
            logger.exception("Error converting SHAP values, using default explanation")
            shap_explanation = [ShapValue(**sv) for sv in self._default_shap_values()]
        
       
        factors = {
            "toss": f"Won by {match_data.toss_winner}" if match_data.toss_winner else "N/A",
            "toss_decision": match_data.toss_decision if match_data.toss_decision else "N/A",
            "venue": match_data.venue,
            "match_type": match_data.match_type
        }
        
        return PredictionResponse(
            winner=winner,
            probability=round(batting_win_prob, 2),
            confidence=confidence,
            shap_explanation=shap_explanation,
            factors=factors
        )
    
    def _generate_dynamic_shap_values(self, model_input: dict) -> List[dict]:
       
        import random
        
       
        runs_required = model_input.get('runs_required', 150)
        wickets_in_hand = model_input.get('wickets_in_hand', 10)
        balls_remaining = model_input.get('balls_remaining', 120)
        required_run_rate = model_input.get('required_run_rate', 7.5)
        current_run_rate = model_input.get('current_run_rate', 6.0)
        
       
        shap_values = []
        
        
        runs_impact = -0.05 - (runs_required / 200) * 0.2 + random.uniform(-0.03, 0.03)
        shap_values.append({
            'feature': 'Runs Required',
            'value': round(runs_impact, 3),
            'impact': 'negative' if runs_impact < 0 else 'positive'
        })
        
        
        wickets_impact = (wickets_in_hand / 10) * 0.25 + random.uniform(-0.02, 0.02)
        shap_values.append({
            'feature': 'Wickets In Hand',
            'value': round(wickets_impact, 3),
            'impact': 'positive' if wickets_impact > 0 else 'negative'
        })
        
        
        rrr_impact = -0.05 - (max(0, required_run_rate - 6) / 10) * 0.3 + random.uniform(-0.02, 0.02)
        shap_values.append({
            'feature': 'Required Run Rate',
            'value': round(rrr_impact, 3),
            'impact': 'negative' if rrr_impact < 0 else 'positive'
        })
        
        
        crr_impact = (current_run_rate / 10) * 0.2 + random.uniform(-0.02, 0.02)
        shap_values.append({
            'feature': 'Current Run Rate',
            'value': round(crr_impact, 3),
            'impact': 'positive' if crr_impact > 0 else 'negative'
        })
        
       
        balls_impact = (balls_remaining / 120) * 0.15 + random.uniform(-0.02, 0.02)
        shap_values.append({
            'feature': 'Balls Remaining',
            'value': round(balls_impact, 3),
            'impact': 'positive' if balls_impact > 0 else 'negative'
        })
        
        
        shap_values = sorted(shap_values, key=lambda x: abs(x['value']), reverse=True)
        
        return shap_values[:5]  # Return top 5
    
    def _default_shap_values(self) -> List[dict]:
        """
        Default SHAP values when model is not available (static fallback)
        """
        return [
            {'feature': 'Team Strength', 'value': 0.15, 'impact': 'positive'},
            {'feature': 'Home Advantage', 'value': 0.12, 'impact': 'positive'},
            {'feature': 'Recent Form', 'value': 0.10, 'impact': 'positive'},
            {'feature': 'Toss Impact', 'value': -0.05, 'impact': 'negative'},
            {'feature': 'Venue History', 'value': 0.08, 'impact': 'positive'},
        ]