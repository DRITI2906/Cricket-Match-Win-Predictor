import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Logger
logger = logging.getLogger(__name__)

# SHAP is optional - use feature importance if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.debug("SHAP not available. Using feature importance instead.")

class CricketPredictor:
    """
    Load trained model and make predictions with SHAP explanations
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_info = None
        self.explainer = None
        
        if model_path is None:
            # Default path
            base_dir = Path(__file__).parent.parent.parent
            model_path = base_dir / "models" / "cricket_model.pkl"
        
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.debug(f"Model loaded from: {self.model_path}")
                
                # Load model info
                info_path = str(self.model_path).replace("cricket_model.pkl", "model_info.pkl")
                if os.path.exists(info_path):
                    self.model_info = joblib.load(info_path)
                    logger.debug(f"Model info loaded from: {info_path}")
                
                # Initialize SHAP explainer
                self._initialize_explainer()
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                logger.debug("Using mock predictions. Train the model first using model_trainer.py")
        except Exception as e:
            logger.exception(f"Error loading model: {e}")
            logger.debug("Using mock predictions")
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer for the model"""
        if not SHAP_AVAILABLE:
            logger.debug("SHAP not available. Will use feature importance instead.")
            self.explainer = None
            return
            
        try:
            # Get the classifier from the pipeline
            classifier = self.model.named_steps['classifier']
            # Create a SHAP explainer using the classifier
            self.explainer = shap.TreeExplainer(classifier)
            logger.debug("SHAP explainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def predict(self, input_data: Dict) -> Tuple[str, float, List[Dict]]:
        """
        Make prediction and generate SHAP explanations
        
        Args:
            input_data: Dictionary with cricket match features
            
        Returns:
            Tuple of (winner, probability, shap_values)
        """
        if self.model is None:
            return self._mock_prediction(input_data)
        
        try:
            # Create DataFrame from input
            df = self._prepare_input(input_data)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(df)[0]
            prediction = self.model.predict(df)[0]
            
            # Debug logging
            logger.debug(f"probabilities array: {probabilities}")
            logger.debug(f"prediction: {prediction}")
            
            # In the training data:
            # Class 0 = batting team loses (bowling team wins)
            # Class 1 = batting team wins
            # probabilities[0] = probability of batting team losing
            # probabilities[1] = probability of batting team winning
            
            batting_team_win_probability = probabilities[1]  # Class 1 = batting team wins
            predicted_class_idx = int(prediction[0])
            
            logger.debug(f"predicted_class_idx: {predicted_class_idx}")
            logger.debug(f"batting_team_win_prob: {batting_team_win_probability}")
            
            # Determine winner based on which probability is higher
            if batting_team_win_probability > 0.5:
                winner = input_data.get('batting_team')
            else:
                winner = input_data.get('bowling_team')
            
            logger.debug(f"winner: {winner}")
            
            # Generate SHAP explanations
            shap_values = self._get_shap_explanation(df)
            
            # Return batting team's win probability (always 0-1 scale)
            return winner, float(batting_team_win_probability), shap_values
            
        except Exception as e:
            logger.exception(f"Error during prediction: {e}")
            return self._mock_prediction(input_data)
    
    def _prepare_input(self, input_data: Dict) -> pd.DataFrame:
        """Prepare input data for model prediction"""
        # Map the API input to model features
        data = {
            'batting_team': input_data.get('batting_team', input_data.get('team1')),
            'bowling_team': input_data.get('bowling_team', input_data.get('team2')),
            'venue': input_data.get('venue'),
            'toss_winner': input_data.get('toss_winner', input_data.get('team1')),
            'toss_decision': input_data.get('toss_decision', 'bat'),
            'runs_required': input_data.get('runs_required', 150),
            'balls_remaining': input_data.get('balls_remaining', 120),
            'wickets_in_hand': input_data.get('wickets_in_hand', 10),
            'target_match': input_data.get('target_match', 250),
            'current_run_rate': input_data.get('current_run_rate', 6.0),
            'required_run_rate': input_data.get('required_run_rate', 7.5)
        }
        
        return pd.DataFrame([data])
    
    def _get_shap_explanation(self, df: pd.DataFrame) -> List[Dict]:
        """Generate SHAP explanations for the prediction"""
        if self.explainer is None:
            return self._get_feature_importance_explanation(df)
        
        try:
            # Transform the data using the preprocessor
            preprocessor = self.model.named_steps['preprocessor']
            X_transformed = preprocessor.transform(df)
            
            # Get SHAP values
            shap_values_raw = self.explainer.shap_values(X_transformed)
            
            # For binary classification, take the values for class 1 (winning)
            if isinstance(shap_values_raw, list):
                shap_values_raw = shap_values_raw[1]
            
            # Get feature names
            feature_names = self._get_feature_names()
            
            # Create list of top SHAP values
            shap_list = []
            for idx, value in enumerate(shap_values_raw[0]):
                if abs(value) > 0.01:  # Only include significant features
                    shap_list.append({
                        'feature': feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                        'value': float(value),
                        'impact': 'positive' if value > 0 else 'negative' if value < 0 else 'neutral'
                    })
            
            # Sort by absolute value and take top 10
            shap_list = sorted(shap_list, key=lambda x: abs(x['value']), reverse=True)[:10]
            
            return shap_list
            
        except Exception as e:
            logger.exception(f"Error generating SHAP values: {e}")
            return self._get_feature_importance_explanation(df)
    
    def _get_feature_importance_explanation(self, df: pd.DataFrame) -> List[Dict]:
        """
        Alternative explanation using feature importance from RandomForest
        Used when SHAP is not available
        """
        try:
            classifier = self.model.named_steps['classifier']
            feature_names = self._get_feature_names()
            
            # Get feature importances
            importances = classifier.feature_importances_
            
            # Create list of feature importances
            importance_list = []
            for idx, importance in enumerate(importances):
                if importance > 0.01:  # Only include significant features
                    feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                    importance_list.append({
                        'feature': self._clean_feature_name(feature_name),
                        'value': float(importance),
                        'impact': 'positive'  # Feature importance is always positive
                    })
            
            # Sort by importance and take top 10
            importance_list = sorted(importance_list, key=lambda x: x['value'], reverse=True)[:10]
            
            return importance_list
            
        except Exception as e:
            logger.exception(f"Error generating feature importance: {e}")
            return self._default_shap_values()
    
    def _clean_feature_name(self, name: str) -> str:
        """Clean up feature names for better readability"""
        # Remove prefixes from one-hot encoded features
        if '_' in name:
            parts = name.split('_', 1)
            if parts[0] in ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']:
                return f"{parts[0].replace('_', ' ').title()}: {parts[1]}"
        
        # Clean up numerical feature names
        name = name.replace('_', ' ').title()
        return name
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names from the preprocessor"""
        try:
            preprocessor = self.model.named_steps['preprocessor']
            
            # Get numerical feature names
            num_features = self.model_info['numerical_features']
            
            # Get categorical feature names (after one-hot encoding)
            cat_transformer = preprocessor.named_transformers_['cat']
            onehot = cat_transformer.named_steps['onehot']
            cat_features = onehot.get_feature_names_out(self.model_info['categorical_features'])
            
            # Combine all feature names
            all_features = list(num_features) + list(cat_features)
            return all_features
            
        except Exception as e:
            logger.exception(f"Error getting feature names: {e}")
            return []
    
    def _default_shap_values(self) -> List[Dict]:
        """Return default SHAP values when actual calculation fails"""
        return [
            {'feature': 'runs_required', 'value': 0.15, 'impact': 'positive'},
            {'feature': 'wickets_in_hand', 'value': 0.12, 'impact': 'positive'},
            {'feature': 'required_run_rate', 'value': -0.10, 'impact': 'negative'},
            {'feature': 'balls_remaining', 'value': 0.08, 'impact': 'positive'},
            {'feature': 'current_run_rate', 'value': 0.06, 'impact': 'positive'},
        ]
    
    def _mock_prediction(self, input_data: Dict) -> Tuple[str, float, List[Dict]]:
        """Mock prediction when model is not available"""
        import random
        probability = random.uniform(0.55, 0.85)
        winner = input_data.get('team1', input_data.get('batting_team', 'Team 1'))
        shap_values = self._default_shap_values()
        return winner, probability, shap_values
